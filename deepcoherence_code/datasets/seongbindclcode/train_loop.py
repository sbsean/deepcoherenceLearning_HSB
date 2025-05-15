# filename: train_loop.py
import torch
from tqdm import tqdm
import random
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import IQDataset
import os
import logging
import argparse
import json
from model import clip_gradients_by_layers

# utils.py에 정의된 함수들을 불러옵니다.
from utils import (
    beamforming,
    create_single_angle_iq,
    compute_cnr_pairwise,  # pairwise 방식 CNR 계산 함수
)


####################################
# 데이터 전처리 및 헬퍼 함수
####################################
def sampling_data(data):
    """
    data: (B,74,2,H,W)
    임의의 각도를 선택하여 x: (B,2,H,W), y: (B, num_angles-1,2,H,W)로 분리
    """
    if isinstance(data, list):
        data = torch.stack(data, dim=0)
    B, num_angles, _, H, W = data.shape
    idx = random.randint(0, num_angles - 1)
    x = data[:, idx]
    y_left = data[:, :idx]
    y_right = data[:, idx + 1 :]
    y = torch.cat((y_left, y_right), dim=1)
    return x, y


def transform_validation_sample(center_x, center_wo_data):
    """
    validation 모드에서 IQ 이미지를 원본 H×W를 1024×1024로 zero-padding한 후 반환.
    반환: (center_x_padded, transformed_center_wo, mask, scale, pad_info)
    """
    H, W = center_x.shape[1], center_x.shape[2]
    target_H, target_W = 1024, 1024
    pad_h_total = target_H - H
    pad_w_total = target_W - W
    pad_top = pad_h_total // 2
    pad_bottom = pad_h_total - pad_top
    pad_left = pad_w_total // 2
    pad_right = pad_w_total - pad_left

    center_x_padded = F.pad(
        center_x.unsqueeze(0),
        (pad_left, pad_right, pad_top, pad_bottom),
        mode="constant",
        value=0,
    ).squeeze(0)

    transformed_angles = []
    for angle in center_wo_data:
        angle_padded = F.pad(
            angle.unsqueeze(0),
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0,
        ).squeeze(0)
        transformed_angles.append(angle_padded)
    transformed_center_wo = torch.stack(transformed_angles, dim=0)

    mask = torch.ones(
        (1, target_H, target_W), dtype=torch.float32, device=center_x.device
    )
    scale = target_H / H
    pad_info = (pad_top, pad_left)
    return center_x_padded, transformed_center_wo, mask, scale, pad_info


####################################
# 평가 함수: ROI 기반 CNR 계산 (Pairwise 방식)
####################################
def evaluate_single_angle(model, device, cyst_rois, xlims, zlims, dx, dz):
    """
    단일 각도 IQ 데이터를 모델에 넣어 빔포밍 후,
    원본 크기의 B-mode 이미지에서,
    빨간 ROI(감쇠된 반지름)와 인접한 파란 ROI(원래 반지름 기반)를 1대1 비교하여
    pairwise CNR의 평균을 계산합니다.
    """
    model.eval()
    with torch.no_grad():
        iq_padded, (origH, origW), pad_info, scale, params = create_single_angle_iq()
        iq_padded = iq_padded.to(device)
        output = model(iq_padded)
        bmode_full = beamforming(output, log_compression=True)[0]
        pad_top, pad_left = pad_info
        pred_beam_img = bmode_full[
            pad_top : pad_top + origH, pad_left : pad_left + origW
        ]

        # cyst_rois (물리 좌표) → 픽셀 좌표 변환 (원래 반지름과 감쇠 반지름 분리)
        roi_coords = []
        for (z_m, x_m), r_m in cyst_rois:
            row_px = (z_m - zlims[0]) / dx
            col_px = (x_m - xlims[0]) / dx
            orig_rad_px = r_m / dx  # blue ROI 계산용 (원래 값)
            disp_rad_px = orig_rad_px * 0.3  # red ROI 통계/표시용 (감쇠된 값)
            roi_coords.append((row_px, col_px, orig_rad_px, disp_rad_px))

        roi_coords_sorted = sorted(roi_coords, key=lambda x: (x[0], x[1]))
        num_red = len(roi_coords_sorted)
        num_rows = int(math.sqrt(num_red)) if num_red > 0 else 1
        roi_rows = [
            roi_coords_sorted[i * num_rows : (i + 1) * num_rows]
            for i in range(num_rows)
        ]

        # 파란 ROI: 각 row에서 인접한 빨간 ROI 사이 영역 (원래 반지름을 기준으로 계산)
        blue_coords = []
        for row in roi_rows:
            for i in range(len(row) - 1):
                r1, c1, orig_rad1, _ = row[i]
                r2, c2, orig_rad2, _ = row[i + 1]
                dist = math.sqrt((r2 - r1) ** 2 + (c2 - c1) ** 2)
                gap = dist - (orig_rad1 + orig_rad2)
                if gap <= 0:
                    continue
                center_r = (r1 + r2) / 2.0
                center_c = (c1 + c2) / 2.0
                blue_rad = (gap / 2.0) * 0.6
                blue_coords.append((center_r, center_c, blue_rad))

        # pairwise CNR 계산 (compute_cnr_pairwise는 red ROI의 통계 계산에 display 반지름 사용)
        cnr_value = compute_cnr_pairwise(pred_beam_img, roi_coords_sorted, blue_coords)

    return cnr_value


def smooth_curve(data, window=15):
    """주어진 데이터에 대해 이동평균 smoothing 적용"""
    data = np.array(data)
    if len(data) < window:
        return data
    # 앞부분은 누적 평균, 이후는 window 단위 이동평균 계산
    prefix = np.array([np.mean(data[: i + 1]) for i in range(window - 1)])
    conv = np.convolve(data, np.ones(window) / window, mode="valid")
    return np.concatenate([prefix, conv])

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("training.log")],
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """커맨드 라인 인자 파싱"""
    parser = argparse.ArgumentParser(description="DCLUNet 모델 학습")

    # 학습 설정
    parser.add_argument(
        "--num_epochs", type=int, default=202532, help="총 학습 에포크 수"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="학습 배치 크기")
    parser.add_argument("--val_batch_size", type=int, default=1, help="검증 배치 크기")
    parser.add_argument(
        "--accumulation_steps", type=int, default=1, help="그래디언트 누적 단계"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="학습률")
    parser.add_argument(
        "--t0_steps", type=int, default=20000, help="코사인 스케줄러 T0 파라미터"
    )
    parser.add_argument(
        "--eta_min", type=float, default=1e-7, help="코사인 스케줄러 최소 학습률"
    )

    # 경로 설정
    parser.add_argument(
        "--base_dir",
        type=str,
        default=r"C:\Users\seongbin\workspace\deepcoherenceLearning_HSB\cubdl\datasets\seongbindclcode\beamformed_data",
        help="데이터 기본 디렉토리",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=r"C:\Users\seongbin\workspace\deepcoherenceLearning_HSB\cubdl\datasets\seongbindclcode\model_result",
        help="모델 저장 디렉토리",
    )

    # 데이터 관련 설정
    parser.add_argument(
        "--folders",
        nargs="+",
        default=[
            "PICMUS",
            "TSH",
            "MYO",
            "JHU",
            "UFL",
            "INS",
            "OSL",
        ],  # ["PICMUS", "TSH", "MYO", "JHU", "UFL", "INS"]
        # ["OSL", "TSH", "MYO", "UFL", "EUT", "INS", "PICMUS"],
        help="데이터 폴더 리스트",
    )

    # 물리 좌표 파라미터
    parser.add_argument(
        "--xlims", nargs=2, type=float, default=[-0.01905, 0.01905], help="x 축 범위"
    )
    parser.add_argument(
        "--zlims", nargs=2, type=float, default=[0e-3, 42e-3], help="z 축 범위"
    )
    parser.add_argument("--dx", type=float, default=0.1e-3, help="x 축 해상도")
    parser.add_argument("--dz", type=float, default=0.1e-3, help="z 축 해상도")

    # 기타 설정
    parser.add_argument(
        "--plot_interval", type=int, default=5000, help="그래프 저장 간격(에포크)"
    )
    parser.add_argument("--gpu", type=int, default=0, help="사용할 GPU 인덱스")
    parser.add_argument(
        "--model_suffix", type=str, default="kaiming", help="모델 파일명 접미사"
    )

    return parser.parse_args()


def save_training_graphs(
    save_dir, args, train_loss_window, val_loss_steps, val_cnr_steps, lr_steps
):
    """학습 과정을 시각화하는 그래프 저장"""
    try:
        # 학습 손실 그래프
        if len(train_loss_window) > 0:
            smoothing_window_train = min(15, len(train_loss_window))
            valid_train_losses = [
                x for x in train_loss_window if not np.isnan(x) and not np.isinf(x)
            ]
            valid_steps = np.arange(1, len(valid_train_losses) + 1)
            if len(valid_train_losses) > 0:
                train_loss_smoothed = smooth_curve(
                    valid_train_losses, window=smoothing_window_train
                )
                plt.figure(figsize=(8, 5))
                plt.plot(
                    valid_steps,
                    valid_train_losses,
                    marker="o",
                    label="Train Loss (raw)",
                    alpha=0.3,
                )
                plt.plot(
                    valid_steps,
                    train_loss_smoothed,
                    label="Train Loss (smoothed)",
                    color="red",
                )
                plt.xlabel("Batch Index")
                plt.ylabel("Train Loss")
                plt.title("Train Loss (Recent Window)")
                plt.legend()
                plt.tight_layout()
                plt.savefig(
                    os.path.join(save_dir, f"train_loss_{args.model_suffix}.png")
                )
                plt.close()

        # 학습률 그래프
        lr_steps_x = [item[0] for item in lr_steps]
        lr_steps_y = [item[1] for item in lr_steps]
        plt.figure(figsize=(8, 5))
        plt.plot(lr_steps_x, lr_steps_y)
        plt.xlabel("Global Step")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"lr_schedule_{args.model_suffix}.png"))
        plt.close()

        # 검증 손실 그래프
        val_steps_plot = [item[0] for item in val_loss_steps]
        val_loss_data = [item[1] for item in val_loss_steps]
        valid_indices = [
            i
            for i, x in enumerate(val_loss_data)
            if not np.isnan(x) and not np.isinf(x)
        ]
        valid_val_steps = [val_steps_plot[i] for i in valid_indices]
        valid_val_loss = [val_loss_data[i] for i in valid_indices]

        if len(valid_val_loss) > 0:
            val_loss_smoothed = smooth_curve(
                valid_val_loss, window=min(15, len(valid_val_loss))
            )
            plt.figure(figsize=(8, 5))
            plt.plot(
                valid_val_steps,
                valid_val_loss,
                marker="o",
                label="Val Loss (raw)",
                alpha=0.3,
            )
            plt.plot(
                valid_val_steps,
                val_loss_smoothed,
                label="Val Loss (smoothed)",
                color="red",
            )
            plt.xlabel("Global Step")
            plt.ylabel("Validation Loss")
            plt.title("Validation Loss per Epoch")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"val_loss_{args.model_suffix}.png"))
            plt.close()

        # CNR 그래프
        val_cnr_data = [item[1] for item in val_cnr_steps]
        valid_indices = [
            i for i, x in enumerate(val_cnr_data) if not np.isnan(x) and not np.isinf(x)
        ]
        valid_cnr_steps_plot = [val_steps_plot[i] for i in valid_indices]
        valid_val_cnr = [val_cnr_data[i] for i in valid_indices]

        if len(valid_val_cnr) > 0:
            val_cnr_smoothed = smooth_curve(
                valid_val_cnr, window=min(15, len(valid_val_cnr))
            )
            plt.figure(figsize=(8, 5))
            plt.plot(
                valid_cnr_steps_plot,
                valid_val_cnr,
                marker="o",
                label="Val CNR (raw)",
                alpha=0.3,
            )
            plt.plot(
                valid_cnr_steps_plot,
                val_cnr_smoothed,
                label="Val CNR (smoothed)",
                color="red",
            )
            plt.xlabel("Global Step")
            plt.ylabel("CNR (dB)")
            plt.title("Validation CNR per Epoch")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"val_cnr_{args.model_suffix}.png"))
            plt.close()

    except Exception as e:
        logger.error(f"그래프 생성 중 예외 발생: {str(e)}")


def save_checkpoint(model, save_path, info=None):
    """모델 체크포인트 저장"""
    try:
        checkpoint = {
            "model_state_dict": model.state_dict(),
        }
        if info:
            checkpoint["info"] = info

        torch.save(checkpoint, save_path)
        logger.info(f"모델 저장 완료: {save_path}")
        return True
    except Exception as e:
        logger.error(f"모델 저장 중 오류 발생: {str(e)}")
        return False


def load_checkpoint(model, load_path, device):
    """모델 체크포인트 로드"""
    try:
        checkpoint = torch.load(load_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"모델 로드 완료: {load_path}")
        info = checkpoint.get("info", None)
        return model, info
    except Exception as e:
        logger.error(f"모델 로드 중 오류 발생: {str(e)}")
        return model, None


def prepare_validation_data(candidate_paths, filelist_val, val_batch_size):
    """검증 데이터셋 및 데이터로더 준비"""
    val_file_paths = []
    try:
        for source, candidate_list in filelist_val.items():
            for key, path in candidate_paths.items():
                if (
                    isinstance(key, tuple)
                    and key[0] == source
                    and key[1] in candidate_list
                ):
                    val_file_paths.append(path)

        val_dataset = IQDataset(
            val_file_paths, train=False, validation=True, inference=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        return val_loader
    except Exception as e:
        logger.error(f"검증 데이터 준비 중 오류 발생: {str(e)}")
        raise


def prepare_training_data(candidate_paths, filelist_train, batch_size):
    """학습 데이터셋 및 데이터로더 준비"""
    train_file_paths = []
    try:
        for source, candidate_list in filelist_train.items():
            for key, path in candidate_paths.items():
                if (
                    isinstance(key, tuple)
                    and key[0] == source
                    and key[1] in candidate_list
                ):
                    train_file_paths.append(path)

        train_dataset = IQDataset(
            train_file_paths, train=True, validation=False, inference=False
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        return train_loader
    except Exception as e:
        logger.error(f"학습 데이터 준비 중 오류 발생: {str(e)}")
        raise


def train_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    accumulation_steps,
    global_step,
):
    model.train()
    running_loss = 0.0
    train_loss_window = []
    lr_steps = []

    batch_pbar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Training",
        position=1,
        leave=False,
    )

    for i, batch in batch_pbar:
        try:
            batch = batch.to(device)
            B, num_angles, C, H, W = batch.shape
            idx = torch.randint(0, num_angles, (1,)).item()
            x = batch[:, idx]
            y_left = batch[:, :idx]
            y_right = batch[:, idx + 1 :]
            y = torch.cat((y_left, y_right), dim=1)

            output = model(x)
            sample_losses = criterion(output, y)  # (B,)

            # 샘플별 backward (accumulation 고려)
            for sample_loss in sample_losses:
                (sample_loss / accumulation_steps).backward()

            if (i + 1) % accumulation_steps == 0:
                clip_gradients_by_layers(
                    model,
                    encoder_clip_value=1.2,
                    decoder_clip_value=1.2,
                    final_clip_value=0.7,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            scheduler.step(global_step)
            current_lr = optimizer.param_groups[0]["lr"]
            lr_steps.append((global_step, current_lr))
            batch_loss = sample_losses.mean().item()
            running_loss += batch_loss

            if len(train_loss_window) >= 1000:
                train_loss_window.pop(0)
            train_loss_window.append(batch_loss)

            global_step += 1

            batch_pbar.set_postfix(
                {
                    "LR": f"{current_lr:.7f}",
                    "Train Loss": f"{batch_loss:.4f}",
                }
            )
        except Exception as e:
            logger.error(f"배치 {i} 학습 중 오류 발생: {str(e)}")
            continue

    avg_loss = (
        running_loss / len(train_loader) if len(train_loader) > 0 else float("nan")
    )
    return avg_loss, global_step, train_loss_window, lr_steps


def validate(model, val_loader, criterion, device, cyst_rois, xlims, zlims, dx, dz):
    """모델 검증 수행 (각 스텝마다 CNR 측정 및 리스트 반환)"""
    model.eval()
    val_loss = 0.0
    val_batches = 0
    cnr_values = []

    try:
        with torch.no_grad():
            for val_batch in val_loader:
                # 입력·출력 분리 및 손실 계산
                val_batch = val_batch.to(device)
                B_val, num_angles_val, C_val, H_val, W_val = val_batch.shape
                idx_val = 0
                x_val = val_batch[:, idx_val]
                y_left_val = val_batch[:, :idx_val]
                y_right_val = val_batch[:, idx_val + 1 :]
                y_val = torch.cat((y_left_val, y_right_val), dim=1)

                output_val = model(x_val)
                sample_losses_val = criterion(output_val, y_val)
                val_loss += sample_losses_val.mean().item()
                val_batches += 1

                # -- 각 배치마다 CNR 측정 --
                cnr_step = evaluate_single_angle(
                    model, device, cyst_rois, xlims, zlims, dx, dz
                )
                if torch.is_tensor(cnr_step):
                    cnr_step = cnr_step.item()
                cnr_values.append(cnr_step)
                # 디버깅용 출력 (필요 시 활성화)
                # print(f"[Validation] Step {val_batches}: CNR = {cnr_step:.4f}")

        # 평균 손실 계산
        avg_val_loss = val_loss / val_batches if val_batches > 0 else float("nan")

        # 이제 평균 CNR은 호출부에서 원하는 방식으로 계산할 수 있습니다.
        return avg_val_loss, cnr_values

    except Exception as e:
        logger.error(f"검증 중 오류 발생: {str(e)}")
        return float("nan"), []


def save_training_config(args, save_dir):
    """학습 설정 저장"""
    config_path = os.path.join(save_dir, f"training_config_{args.model_suffix}.json")
    try:
        with open(config_path, "w") as f:
            json.dump(vars(args), f, indent=4)
        logger.info(f"학습 설정 저장 완료: {config_path}")
    except Exception as e:
        logger.error(f"학습 설정 저장 중 오류 발생: {str(e)}")
