# File: data_preprocess_cubdl.py
# Author: [Your Name]
# Description: cubdl task1 데이터에 대해 각 angle별 DAS 결과를 beamforming하여,
#              정규화 없이 각 data_source별 지정된 경로에 pt 파일로 저장

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import shutil

try:
    from cubdl.das_torch import DAS_PW
    from cubdl.PixelGrid import make_pixel_grid
    from datasets.PWDataLoaders import load_data
except ImportError:
    print("[ERROR] cubdl 또는 PWDataLoaders 임포트가 불가능합니다. 환경을 확인하세요.")
    raise


def initialize_cubdl_output(initialize_flag=True):
    base_save_dir = r"C:\Users\seongbin\workspace\deepcoherenceLearning_HSB\cubdl\datasets\seongbindclcode\beamformed_data"
    if initialize_flag and os.path.exists(base_save_dir):
        shutil.rmtree(base_save_dir)
        print(f"[INFO] 기존 cubdl 출력 폴더 '{base_save_dir}' 삭제됨.")
    else:
        print("[INFO] cubdl 초기화 건너뜀.")


def beamform_cubdl_candidates():
    """
    각 angle별 DAS 결과를 계산하고,
    정규화 없이 각 data_source별로 pt 파일로 저장.
    """
    candidate_dict = {}
    big_filelist = {
        "JHU": list(range(24, 35)),
        "INS": list(range(1, 27)),
        "MYO": [1, 2, 3, 4, 5, 6],
        "UFL": [1, 2, 4, 5],
        "OSL": [7, 10],
        "EUT": [3, 6],
        "TSH": list(range(2, 502)),
    }

    for data_source, acqs in big_filelist.items():
        for acq in acqs:
            print(f"[INFO] Beamforming cubdl: {data_source}, acq={acq}")
            P, xlims, zlims = load_data(data_source, acq)
            wvln = P.c / P.fc
            dx = wvln / 2.5
            dz = dx
            grid = make_pixel_grid(xlims, zlims, dx, dz)

            # beamforming 수행 및 angle별 리스트 생성
            num_angles = P.angles.shape[0]
            angle_list = []
            for idx_angle in range(num_angles):
                das_op = DAS_PW(P, grid, idx_angle, rxfnum=1)
                idas, qdas = das_op((P.idata, P.qdata))
                iq_tensor = torch.stack([idas, qdas], dim=0)  # (2, H, W)
                angle_list.append(iq_tensor)

            # JHU 데이터 중 특정 acq만 0도(angle 0) 위치 조정: 원래 인덱스 0을 중앙 위치로 이동
            if data_source == "JHU" and acq in [28, 29, 31, 32, 33, 34]:
                zero_angle = angle_list.pop(0)
                mid_idx = num_angles // 2
                angle_list.insert(mid_idx, zero_angle)
            # 스택하여 (k,2,H,W) 텐서 생성
            angle_data = torch.stack(angle_list, dim=0)
            candidate_dict[(data_source, acq)] = angle_data.cpu()

    # 저장 디렉토리 구성 및 파일 쓰기
    base_save_dir = r"C:\Users\seongbin\workspace\deepcoherenceLearning_HSB\cubdl\datasets\seongbindclcode\beamformed_data"
    for (data_source, acq), candidate in candidate_dict.items():
        save_dir = os.path.join(base_save_dir, data_source)
        os.makedirs(save_dir, exist_ok=True)
        pt_filename = os.path.join(save_dir, f"one_angle_tensor_{acq}.pt")
        torch.save(candidate, pt_filename)
        print(f"[INFO] Saved candidate for {data_source}, acq={acq} to {pt_filename}")

    overall_filename = os.path.join(base_save_dir, "result_dict.pt")
    torch.save(candidate_dict, overall_filename)
    print(f"[INFO] Saved overall result dictionary to {overall_filename}")

    # Optional: 데이터 확인용 plotting (정규화 없이 저장된 결과 확인)
    for key, data in candidate_dict.items():
        data_source, acq = key
        center_idx = data.shape[0] // 2
        raw_center_tensor = data[center_idx]
        raw_iq = raw_center_tensor[0].cpu() + 1j * raw_center_tensor[1].cpu()
        raw_bimg = 20 * np.log10(np.abs(raw_iq.numpy()))
        raw_bimg -= np.amax(raw_bimg)
        plt.figure(figsize=(6, 6))
        plt.imshow(raw_bimg, cmap="gray", origin="upper", vmin=-60)
        plt.title(f"Raw DAS (No Norm) - {data_source} {acq}")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    initialize_cubdl_output(initialize_flag=False)
    beamform_cubdl_candidates()
