import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from cubdl.das_torch import DAS_PW
from cubdl.PlaneWaveData import (
    PICMUSData,
    InVivoPICMUSData,
)
from cubdl.PixelGrid import make_pixel_grid
import shutil

print("Current working directory:", os.getcwd())

database_path = os.path.join("cubdl", "datasets", "data", "picmus", "database")
acq_list = ["simulation", "experiments", "in_vivo"]
target_list = ["contrast_speckle", "resolution_distorsion"]
dtype = "iq"

invivo_files = [
    ("carotid_cross", "carotid_cross_expe_dataset_iq.hdf5"),
    ("carotid_long", "carotid_long_expe_dataset_iq.hdf5"),
]

result_dict = {}
picmus_save_dir = r"C:\Users\seongbin\workspace\deepcoherenceLearning_HSB\cubdl\datasets\seongbindclcode\beamformed_data\PICMUS"


def initialize_picmus_output(initialize_flag=False):
    if initialize_flag and os.path.exists(picmus_save_dir):
        shutil.rmtree(picmus_save_dir)
        print(f"[INFO] 이전 PICMUS 출력 폴더 '{picmus_save_dir}' 삭제됨.")
    else:
        print("[INFO] PICMUS 초기화 건너뜀.")
    os.makedirs(picmus_save_dir, exist_ok=True)


initialize_picmus_output(initialize_flag=True)

# ================================
# ✅ 1. simulation / experiments 먼저 처리 (PICMUSData)
# ================================
for acq in ["simulation", "experiments"]:
    for target in target_list:
        print(f"Processing: {acq}/{target}")
        P = PICMUSData(database_path, acq, target, dtype)
        xlims = [P.ele_pos[0, 0], P.ele_pos[-1, 0]]
        zlims = [5e-3, 50e-3] if acq == "simulation" else [0e-3, 42e-3]

        wvln = P.c / P.fc
        dx = dz = wvln / 3
        grid = make_pixel_grid(xlims, zlims, dx, dz)
        iqdata = (P.idata, P.qdata)

        angle_tensor_list = []
        for i in range(len(P.angles)):
            das_angle = DAS_PW(P, grid, i)
            idas, qdas = das_angle(iqdata)
            if idas.dim() == 2:
                idas = idas.unsqueeze(0)
            if qdas.dim() == 2:
                qdas = qdas.unsqueeze(0)
            angle_tensor_list.append(torch.cat([idas, qdas], dim=0))

        tensor = torch.stack(angle_tensor_list, dim=0)
        result_key = (target, acq)
        result_dict[result_key] = tensor

        pt_file = os.path.join(picmus_save_dir, f"one_angle_tensor_{acq}_{target}.pt")
        torch.save(tensor, pt_file)
        print(f"[INFO] Saved tensor: {pt_file}")

        mid = tensor.shape[0] // 2
        img = tensor[mid]
        iq = img[0].cpu() + 1j * img[1].cpu()
        bimg = 20 * np.log10(np.abs(iq.numpy()))
        bimg -= np.max(bimg)
        plt.figure(figsize=(6, 6))
        plt.imshow(bimg, cmap="gray", origin="upper", vmin=-60)
        plt.title(f"DAS (No Norm) - {target} {acq}")
        plt.axis("off")
        plt.show()

# ================================
# ✅ 2. in_vivo 처리 (InVivoPICMUSData)
# ================================
for subdir, filename in invivo_files:
    print(f"Processing: in_vivo/{subdir}")
    full_path = os.path.join(database_path, "in_vivo", subdir, filename)
    if not os.path.exists(full_path):
        print(f"[WARNING] {full_path} not found. Skipping.")
        continue

    P = InVivoPICMUSData(filepath=full_path, dtype="iq")
    xlims = [P.ele_pos[0, 0], P.ele_pos[-1, 0]]
    zlims = [0e-3, 42e-3]

    wvln = P.c / P.fc
    dx = dz = wvln / 3
    grid = make_pixel_grid(xlims, zlims, dx, dz)
    iqdata = (P.idata, P.qdata)

    angle_tensor_list = []
    for i in range(len(P.angles)):
        das_angle = DAS_PW(P, grid, i)
        idas, qdas = das_angle(iqdata)
        if idas.dim() == 2:
            idas = idas.unsqueeze(0)
        if qdas.dim() == 2:
            qdas = qdas.unsqueeze(0)
        angle_tensor_list.append(torch.cat([idas, qdas], dim=0))

    tensor = torch.stack(angle_tensor_list, dim=0)
    result_key = (subdir, "in_vivo")
    result_dict[result_key] = tensor

    pt_file = os.path.join(picmus_save_dir, f"one_angle_tensor_in_vivo_{subdir}.pt")
    torch.save(tensor, pt_file)
    print(f"[INFO] Saved tensor: {pt_file}")

    mid = tensor.shape[0] // 2
    img = tensor[mid]
    iq = img[0].cpu() + 1j * img[1].cpu()
    bimg = 20 * np.log10(np.abs(iq.numpy()))
    bimg -= np.max(bimg)
    plt.figure(figsize=(6, 6))
    plt.imshow(bimg, cmap="gray", origin="upper", vmin=-60)
    plt.title(f"In-Vivo DAS (No Norm) - {subdir}")
    plt.axis("off")
    plt.show()

# ✅ 전체 저장
overall_path = os.path.join(picmus_save_dir, "result_dict.pt")
torch.save(result_dict, overall_path)
print(f"[INFO] ✅ All processing done. Results saved to {overall_path}")
