# filename: main.py

import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import IQDataset
from model import DCLUNet, init_weights
from loss_fn import CoherenceLoss
from utils import load_cyst_info

import train_loop
from datasets.PWDataLoaders import get_filelist
from candidate_paths import load_candidate_paths_from_folders

logger = train_loop.logger


def main():
    args = train_loop.parse_arguments()

    # lr settings
    args.lr = 1e-4
    args.eta_min = 1e-7

    start_time = time.time()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loop.save_training_config(args, save_dir)

    logger.info("학습 설정:")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"  {arg}: {value}")

    T_0_steps = int(args.t0_steps // args.batch_size * args.accumulation_steps)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"사용 디바이스: {device}")

    # model, optimizer
    try:
        model = DCLUNet().to(device)
        init_weights(model, init_type="kaiming")

        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0_steps, T_mult=1, eta_min=args.eta_min
        )
        scheduler = cosine_scheduler

        criterion = CoherenceLoss()
        logger.info("Model and optimizer initialization completed")
    except Exception as e:
        logger.error(f"모델 초기화 중 오류 발생: {str(e)}")
        return

    # TensorBoard Settings
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    tb_log_dir = Path("runs") / timestamp
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(
        log_dir=str(tb_log_dir),
        max_queue=10,
        flush_secs=30,
    )
    logger.info(f"TensorBoard 로그가 {tb_log_dir} 에 저장됩니다.")

    global_step = 0
    best_val_loss = float("inf")
    best_cnr = float("-inf")
    best_cnr_only = float("-inf")
    best_cnr_model_val_loss = None

    # 경로 설정
    try:
        cyst_rois = load_cyst_info()
        candidate_paths = load_candidate_paths_from_folders(args.base_dir, args.folders)
        if len(candidate_paths) == 0:
            raise FileNotFoundError("지정된 폴더에서 pt 파일을 찾을 수 없습니다.")

        picmus_found = any(
            isinstance(key, tuple) and key[0] == "PICMUS" for key in candidate_paths
        )
        logger.info(f"PICMUS 파일 로드 여부: {picmus_found}")

        # Validation 데이터 준비
        filelist_val = get_filelist("task2")
        val_loader = train_loop.prepare_validation_data(
            candidate_paths, filelist_val, args.val_batch_size
        )

        # 학습 파일 리스트 준비
        filelist_train = filelist_val.copy()
    except Exception as e:
        logger.error(f"데이터 로드 중 오류 발생: {str(e)}")
        return

    # 모델 파일 경로 설정
    best_model_path = os.path.join(
        save_dir, f"model_best_{args.num_epochs}_{args.model_suffix}.pth"
    )
    best_cnr_model_path = os.path.join(
        save_dir, f"model_best_cnr_{args.num_epochs}_{args.model_suffix}.pth"
    )
    final_path = os.path.join(
        save_dir, f"model_final_{args.num_epochs}_{args.model_suffix}.pth"
    )

    # 에포크 반복
    epoch_pbar = tqdm(range(args.num_epochs), desc="Epochs", position=0)
    epoch_times = []

    for epoch in epoch_pbar:
        epoch_start_time = time.time()

        try:
            train_loader = train_loop.prepare_training_data(
                candidate_paths, filelist_train, args.batch_size
            )

            if epoch == 0:
                steps_per_epoch = len(train_loader)
                logger.info(f"에포크당 배치 수: {steps_per_epoch}")

            # 한 에포크 학습
            avg_train_loss, global_step, epoch_loss_window, epoch_lr_steps = (
                train_loop.train_epoch(
                    model,
                    train_loader,
                    optimizer,
                    scheduler,
                    criterion,
                    device,
                    args.accumulation_steps,
                    global_step,
                )
            )

            # NaN 체크 및 복구
            if np.isnan(avg_train_loss):
                logger.warning(
                    f"Epoch {epoch+1}에서 NaN 손실 발생, 모델 가중치 체크 및 수정 적용..."
                )
                with torch.no_grad():
                    for param in model.parameters():
                        if torch.isnan(param).any() or torch.isinf(param).any():
                            logger.warning(
                                "NaN 또는 Inf 파라미터 발견, 리셋 적용 중..."
                            )
                            param.data = torch.randn_like(param.data) * 0.01

            torch.cuda.empty_cache()

            # 검증 수행
            if not np.isnan(avg_train_loss):
                avg_val_loss, cnr_steps = train_loop.validate(
                    model,
                    val_loader,
                    criterion,
                    device,
                    cyst_rois,
                    args.xlims,
                    args.zlims,
                    args.dx,
                    args.dz,
                )
            else:
                avg_val_loss, cnr_steps = float("nan"), []

            # 에포크별 평균 CNR 계산
            avg_cnr_epoch = (
                sum(cnr_steps) / len(cnr_steps) if cnr_steps else float("nan")
            )

            # 에포크 시간 계산 및 ETA 업데이트
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            epoch_times.append(epoch_duration)
            remaining_epochs = args.num_epochs - (epoch + 1)
            recent_epoch_times = (
                epoch_times[-5:] if len(epoch_times) >= 5 else epoch_times
            )
            avg_epoch_time = sum(recent_epoch_times) / len(recent_epoch_times)
            estimated_remaining_time = avg_epoch_time * remaining_epochs
            elapsed_time = time.time() - start_time
            epoch_pbar.set_postfix(
                {
                    "Elapsed": str(timedelta(seconds=int(elapsed_time))),
                    "Remaining": str(timedelta(seconds=int(estimated_remaining_time))),
                    "ETA": str(
                        timedelta(seconds=int(elapsed_time + estimated_remaining_time))
                    ),
                }
            )

            # 로그 출력
            logger.info(
                f"\nEpoch {epoch+1}/{args.num_epochs} 완료 - {epoch_duration:.2f}초 소요"
            )
            logger.info(f"현재 학습률: {optimizer.param_groups[0]['lr']:.8f}")
            logger.info(f"Epoch {epoch+1} Train Loss (avg) = {avg_train_loss:.4f}")
            logger.info(
                f"Epoch {epoch+1}: Val Loss = {avg_val_loss:.4f}, Avg CNR = {avg_cnr_epoch:.4f}"
            )

            # TensorBoard 기록
            writer.add_scalar("Train/AvgLoss", float(avg_train_loss), global_step)
            if not np.isnan(avg_train_loss):
                writer.add_scalar(
                    "Validation/AvgLoss", float(avg_val_loss), global_step
                )
                writer.add_scalar(
                    "Validation/AvgCNR", float(avg_cnr_epoch), global_step
                )
                for i, cs in enumerate(cnr_steps, 1):
                    writer.add_scalar(
                        "Validation/CNR_step",
                        float(cs),
                        global_step - len(cnr_steps) + i,
                    )
            writer.add_scalar(
                "LearningRate", float(optimizer.param_groups[0]["lr"]), global_step
            )
            if global_step % 500 == 0:
                writer.flush()

            # 최상 모델 저장 (복합 지표)
            if not np.isnan(avg_cnr_epoch):
                if (avg_val_loss < best_val_loss) or (
                    abs(avg_val_loss - best_val_loss) < 1e-12
                    and avg_cnr_epoch > best_cnr
                ):
                    best_val_loss = avg_val_loss
                    best_cnr = avg_cnr_epoch
                    info = {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "val_loss": best_val_loss,
                        "cnr": best_cnr,
                    }
                    train_loop.save_checkpoint(model, best_model_path, info)
                    logger.info(
                        f"[INFO] Best model updated at epoch {epoch+1} "
                        f"(step={global_step}): Val Loss={best_val_loss:.4f}, "
                        f"CNR={best_cnr:.4f}, saved to {best_model_path}"
                    )

                # CNR 기준 모델 저장
                if avg_cnr_epoch > best_cnr_only:
                    best_cnr_only = avg_cnr_epoch
                    best_cnr_model_val_loss = avg_val_loss
                    info = {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "val_loss": best_cnr_model_val_loss,
                        "cnr": best_cnr_only,
                    }
                    train_loop.save_checkpoint(model, best_cnr_model_path, info)
                    logger.info(
                        f"[INFO] Best CNR model updated at epoch {epoch+1} "
                        f"(step={global_step}): Val Loss={avg_val_loss:.4f}, "
                        f"CNR={best_cnr_only:.4f}, saved to {best_cnr_model_path}"
                    )

        except Exception as e:
            logger.error(f"에포크 {epoch+1} 학습 중 오류 발생: {str(e)}")
            continue

    # 최종 모델 저장
    total_time = time.time() - start_time
    logger.info(f"\n총 학습 소요 시간: {str(timedelta(seconds=int(total_time)))}")
    logger.info("Training finished.")

    info = {
        "total_epochs": args.num_epochs,
        "total_steps": global_step,
        "training_time": str(timedelta(seconds=int(total_time))),
    }
    train_loop.save_checkpoint(model, final_path, info)
    logger.info(f"[INFO] Final model saved to '{final_path}'")

    # 최종 결과 요약
    logger.info("\n----- Model Summary -----")
    logger.info(f"총 학습 소요 시간: {str(timedelta(seconds=int(total_time)))}")
    logger.info(f"Composite Best Model: {best_model_path}")
    logger.info(f"    -> Val Loss: {best_val_loss:.4f}, CNR: {best_cnr:.4f}")
    logger.info(f"Best CNR Model: {best_cnr_model_path}")
    logger.info(
        f"    -> Val Loss: {best_cnr_model_val_loss:.4f}, CNR: {best_cnr_only:.4f}"
    )
    logger.info(f"Final Model: {final_path}")

    writer.close()


if __name__ == "__main__":
    main()
