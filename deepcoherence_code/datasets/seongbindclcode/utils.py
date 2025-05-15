import math
import random
import torch
import torch.nn.functional as F
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt  # 디버그 플롯용
import cv2  # 시각화용


def compute_bmode(iq_tensor):
    """
    iq_tensor: (2, H, W) 텐서 (I, Q)
    복소수 변환 후 log-compression을 수행하여 B-mode 이미지를 반환.
    """
    iq = iq_tensor[0] + 1j * iq_tensor[1]
    bimg = 20 * np.log10(np.abs(iq.detach().cpu().numpy()) + 1e-8)
    bimg -= np.amax(bimg)
    return bimg


##############################
# 1. 데이터 전처리 유틸
##############################
def sampling_data(data):
    """
    data: (B,74,2,H,W)
    임의의 각도를 선택하여 x: (B,2,H,W), y: (B, num_angles-1,2,H,W)로 분리
    """
    if isinstance(data, list):
        data = torch.stack(data, dim=0)
    B, num_angles, _, H, W = data.shape
    idx = random.randint(0, num_angles - 1)
    x = data[:, idx]  # (B,2,H,W)
    y_left = data[:, :idx]
    y_right = data[:, idx + 1 :]
    y = torch.cat((y_left, y_right), dim=1)  # (B, num_angles-1,2,H,W)
    return x, y


def resize_to_physical(iq_tensor, orig_dx, orig_dz, target_pixel_length):
    """
    iq_tensor: (2, H, W) – I, Q 데이터
    orig_dx, orig_dz: 원본 데이터의 물리적 픽셀 간격
    target_pixel_length: 목표 단위 픽셀 길이
    """
    _, H, W = iq_tensor.shape
    phys_H = H * orig_dz
    phys_W = W * orig_dx
    target_phys_H = 256 * target_pixel_length
    target_phys_W = 256 * target_pixel_length
    scale_H = target_phys_H / phys_H
    scale_W = target_phys_W / phys_W
    scale = min(scale_H, scale_W)
    new_H = max(1, int(round(H * scale)))
    new_W = max(1, int(round(W * scale)))
    iq_resized = F.interpolate(
        iq_tensor.unsqueeze(0),
        size=(new_H, new_W),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return iq_resized, scale


import torch


def beamforming(
    pred: torch.Tensor, log_compression: bool = True, dynamic_range: float = 60.0
):
    """
    Beamforming with optional log-compression and dynamic-range clipping.

    Args:
        pred: Tensor of shape (B, 2, H, W), where [:,0]은 I, [:,1]은 Q 성분
        log_compression: True이면 dB 스케일로 변환 후 정규화 및 클리핑 수행
        dynamic_range: 최대 동적 범위 (0 ~ -dynamic_range dB)

    Returns:
        NumPy array of shape (B, H, W) containing beamformed images in dB (로그 압축 시)
        또는 linear envelope (비압축 시).
    """
    # I/Q 데이터 분리
    i_data = pred[:, 0, :, :]
    q_data = pred[:, 1, :, :]
    # envelope 계산
    envelope = torch.abs(i_data + 1j * q_data)

    if log_compression:
        # dB 변환 (20·log10)
        log_comp = 20 * torch.log10(envelope + 1e-8)

        # 배치별 최대값을 0dB로 정규화
        B, H, W = log_comp.shape
        max_vals, _ = torch.max(log_comp.view(B, -1), dim=1, keepdim=True)
        max_vals = max_vals.view(B, 1, 1)
        log_comp = log_comp - max_vals

        # dynamic range 클리핑: -dynamic_range ≤ log_comp ≤ 0
        log_comp = torch.clamp(log_comp, min=-abs(dynamic_range), max=0.0)

        return log_comp.detach().cpu().numpy()
    else:
        # 비압축(envelope) 그대로 반환
        return envelope.detach().cpu().numpy()


##############################
# 2. CNR 계산용 함수
##############################
def calculate_cnr(mu_i, mu_o, sigma_i, sigma_o):
    """
    mu_i, mu_o: 각각 ROI의 평균값
    sigma_i, sigma_o: 각각 ROI의 표준편차
    위 값들을 이용해 dB 스케일 CNR 계산.
    """
    diff = abs(mu_i - mu_o)
    denom = math.sqrt(sigma_i**2 + sigma_o**2) + 1e-8
    ratio = diff / denom
    if ratio <= 1e-12:
        ratio = 1e-12
    return 20*np.log10(ratio)


##############################
# 3. Phantom ROI (cyst) 정보
##############################
def load_cyst_info():
    """
    phantom 파일에서 cyst ROI 정보를 읽어옴.
    반환: [((center_z_m, center_x_m), radius_m), ...] (단위: meter)
    """
    database_path = os.path.join("cubdl", "datasets", "data", "picmus", "database")
    phantom_file = os.path.join(
        database_path,
        "simulation",
        "contrast_speckle",
        "contrast_speckle_simu_phantom.hdf5",
    )
    with h5py.File(phantom_file, "r") as f:
        ds = f["US"]["US_DATASET0000"]
        centerX = ds["phantom_occlusionCenterX"][:]  # meter
        centerZ = ds["phantom_occlusionCenterZ"][:]  # meter
        diameter = ds["phantom_occlusionDiameter"][:]  # meter
    cyst_rois = []
    for i in range(len(centerX)):
        z_m = float(centerZ[i])
        x_m = float(centerX[i])
        # 원래 반지름 (diameter/2)를 그대로 반환
        radius_m = float(diameter[i]) / 2.0
        cyst_rois.append(((z_m, x_m), radius_m))
    return cyst_rois


def create_single_angle_iq(debug_plot=False):
    """
    원본 (1,2,H,W)를 받아서 inference 모드에 맞게 1024×1024로 zero-padding 적용.
    반환: (iq_padded, (origH, origW), pad_info, scale, (xlims, zlims, dx, dz))
    """
    from cubdl.PlaneWaveData import PICMUSData
    from cubdl.PixelGrid import make_pixel_grid
    from cubdl.das_torch import DAS_PW

    database_path = os.path.join("cubdl", "datasets", "data", "picmus", "database")
    acq = "simulation"
    target = "contrast_speckle"
    dtype = "iq"
    P = PICMUSData(database_path, acq, target, dtype)

    xlims = [P.ele_pos[0, 0], P.ele_pos[-1, 0]]
    zlims = [0e-3, 42e-3]
    wvln = P.c / P.fc
    dx = wvln / 3
    dz = dx
    grid = make_pixel_grid(xlims, zlims, dx, dz)

    idx = len(P.angles) // 2
    das1 = DAS_PW(P, grid, idx)
    iqdata = (P.idata, P.qdata)
    idas1, qdas1 = das1(iqdata)  # (H,W)
    H, W = idas1.shape

    idas1_np = idas1.detach().cpu().numpy()
    qdas1_np = qdas1.detach().cpu().numpy()
    data_2hw = np.stack([idas1_np, qdas1_np], axis=0)  # (2,H,W)
    data_2hw = np.expand_dims(data_2hw, axis=0)  # (1,2,H,W)
    data_tensor = torch.from_numpy(data_2hw).float()

    # Zero-pad to 1024×1024
    target_H, target_W = 1024, 1024
    pad_h_total = target_H - H
    pad_w_total = target_W - W
    pad_top = pad_h_total // 2
    pad_bottom = pad_h_total - pad_top
    pad_left = pad_w_total // 2
    pad_right = pad_w_total - pad_left
    iq_padded = F.pad(
        data_tensor,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode="constant",
        value=0,
    )
    scale = target_H / H
    pad_info = (pad_top, pad_left)
    return iq_padded, (H, W), pad_info, scale, (xlims, zlims, dx, dz)


#########################################
# ROI 통계 계산용 헬퍼 함수 (CNR 점검)
# --> "중앙값" 대신 평균(mean)과 표준편차(std)로 계산
#########################################
def compute_roi_stats(image, center, radius):
    """
    image: (H,W) B-mode 이미지 (dB scale)
    center: (row, col)
    radius: ROI 반지름 (pixel)

    ROI 내 픽셀들의 평균과 표준편차를 계산하여 반환합니다.
    """
    rows, cols = np.ogrid[: image.shape[0], : image.shape[1]]
    mask = (rows - center[0]) ** 2 + (cols - center[1]) ** 2 <= radius**2
    roi_pixels = image[mask]
    if roi_pixels.size == 0:
        return 0.0, 0.0
    mean_val = np.mean(roi_pixels)
    std_val = np.std(roi_pixels)
    return mean_val, std_val


#########################################
# 새 방식: 인접한 영역 1대 1 비교하여 평균 CNR 계산 함수
#########################################
def compute_cnr_pairwise(pred_beam_img, roi_coords_sorted, blue_coords):
    """
    pred_beam_img: (H,W) B-mode 이미지 (dB scale)
    roi_coords_sorted: 리스트 of red ROI 좌표들, 각 항목은 (row, col, orig_rad, disp_rad) 형태 (총 9개라 가정)
    blue_coords: 리스트 of blue ROI 좌표들, 각 항목은 (row, col, radius) 형태 (총 6개라 가정)

    각 red ROI에 대해, blue_coords 중 가장 가까운 ROI를 찾아 1대1 비교로 CNR을 계산하고,
    그 CNR들의 평균을 반환합니다.
    - red ROI의 통계 계산에는 display radius (감소된 값)을 사용합니다.
    """
    pairwise_cnr = []
    for red_roi in roi_coords_sorted:
        # red_roi: (row, col, orig_rad, disp_rad)
        red_mean, red_std = compute_roi_stats(
            pred_beam_img, (red_roi[0], red_roi[1]), red_roi[3]
        )
        best_dist = None
        best_blue = None
        for blue_roi in blue_coords:
            dist = math.sqrt(
                (red_roi[0] - blue_roi[0]) ** 2 + (red_roi[1] - blue_roi[1]) ** 2
            )
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_blue = blue_roi
        if best_blue is not None:
            blue_mean, blue_std = compute_roi_stats(
                pred_beam_img, (best_blue[0], best_blue[1]), best_blue[2]
            )
            cnr_val = calculate_cnr(red_mean, blue_mean, red_std, blue_std)
            pairwise_cnr.append(cnr_val)
    if pairwise_cnr:
        return np.mean(pairwise_cnr)
    else:
        return 0.0


##############################
# 메인 실행부 (빨간원 9개 + 파란원 6개)
##############################
if __name__ == "__main__":
    # (1) 단일 각도 IQ 생성 (Inference mode: zero-padding to 1024×1024)
    iq_padded, (origH, origW), pad_info, scale, (xlims, zlims, dx, dz) = (
        create_single_angle_iq(debug_plot=False)
    )
    print("Final IQ shape:", iq_padded.shape)  # (1,2,1024,1024)

    # (2) 빔포밍: compute_bmode를 사용하여 log 압축 및 정규화 수행
    beam_img_full = compute_bmode(iq_padded[0])  # (1024,1024) B-mode 이미지
    print(
        f"[INFO] Beamformed image range: min={beam_img_full.min():.2f}, max={beam_img_full.max():.2f}"
    )

    # (3) Zero-padding 영역 제거하여 원본 크기 (origH×origW) 이미지 추출
    pad_top, pad_left = pad_info
    pred_beam_img = beam_img_full[
        pad_top : pad_top + origH, pad_left : pad_left + origW
    ]
    print(
        f"[INFO] Predicted B-mode image size after removing padding: {pred_beam_img.shape}"
    )

    # (4) Cyst ROI 로드 -> 빨간원 (예: 9개, 3행×3열로 가정)
    cyst_rois = load_cyst_info()  # 반환: [((z_m, x_m), r_m), ...] in meters

    # 원본 영상 픽셀 좌표계로 변환 (zero-padding 제거 후의 이미지와 동일한 좌표계)
    # 여기서 각 ROI에 대해 원래 반지름과 display용(0.7배) 반지름을 함께 계산합니다.
    roi_coords = []
    for (z_m, x_m), r_m in cyst_rois:
        row_px = (z_m - zlims[0]) / dx
        col_px = (x_m - xlims[0]) / dx
        orig_rad_px = r_m / dx  # blue ROI 계산용 (원래 값)
        disp_rad_px = orig_rad_px * 0.47 # red ROI 통계/표시용
        roi_coords.append((row_px, col_px, orig_rad_px, disp_rad_px))

    # ROI 좌표 정렬 (row, col 순)
    roi_coords_sorted = sorted(roi_coords, key=lambda x: (x[0], x[1]))
    # 9개 ROI를 3행×3열로 그룹화 (예시)
    roi_rows = [roi_coords_sorted[i * 3 : (i + 1) * 3] for i in range(3)]

    # (5) 인접한 빨간원 사이 영역을 이용해 파란원(배경) 영역 산출 (각 row에서 인접한 두 ROI의 중간 영역)
    blue_coords = []
    for row in roi_rows:
        for i in range(len(row) - 1):
            # blue ROI 계산에는 원래 반지름을 사용합니다.
            r1, c1, orig_rad1, _ = row[i]
            r2, c2, orig_rad2, _ = row[i + 1]
            dist = math.sqrt((r2 - r1) ** 2 + (c2 - c1) ** 2)
            gap = dist - (orig_rad1 + orig_rad2)
            if gap <= 0:
                continue
            center_r = (r1 + r2) / 2.0
            center_c = (c1 + c2) / 2.0
            blue_rad = (gap / 2.0)*0.9
            blue_coords.append((center_r, center_c, blue_rad))

    # (6) ROI 영역별 1대 1 비교를 통한 CNR 산출
    cnr_value = compute_cnr_pairwise(pred_beam_img, roi_coords_sorted, blue_coords)
    print(f"[Pairwise] Calculated CNR: {cnr_value:.2f} dB")

    # (7) 플롯 (선택)
    extent = [0, origW, origH, 0]
    plt.figure(figsize=(6, 6))
    plt.title("Beamformed Image with ROI (Red) & Background (Blue) - Pairwise CNR")
    plt.imshow(pred_beam_img, vmin=-60, cmap="gray", extent=extent, origin="upper")
    ax = plt.gca()
    ax.set_aspect("auto")

    # 빨간원 9개 표시 (표시할 때는 감쇠된 반지름 사용)
    for row_px, col_px, orig_rad, disp_rad in roi_coords_sorted:
        circ_red = plt.Circle(
            (col_px, row_px), disp_rad, color="red", fill=False, linewidth=2
        )
        ax.add_patch(circ_red)

    # 파란원 표시
    for row_b, col_b, rad_b in blue_coords:
        circ_blue = plt.Circle(
            (col_b, row_b), rad_b, color="blue", fill=False, linewidth=2
        )
        ax.add_patch(circ_blue)

    plt.xlabel("col (pixel)")
    plt.ylabel("row (pixel)")
    plt.tight_layout()
    plt.show()
