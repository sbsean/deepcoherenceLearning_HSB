# File: example_picmus_tf.py
# Author: Dongwoon Hyun (dongwoon.hyun@stanford.edu)
# Created on: 2020-04-27
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from cubdl.das_tf import DAS_PW
from cubdl.PlaneWaveData import PICMUSData
from cubdl.PixelGrid import make_pixel_grid

# 현재 작업 디렉터리 확인 (디버깅용)
print("현재 작업 디렉터리:", os.getcwd())

# 절대 경로를 사용하거나, 작업 디렉터리에 맞게 상대 경로를 수정합니다.
# 현재 작업 디렉터리가 "C:\Users\seongbin\workspace\deepcoherenceLearning_HSB"라면,
# 데이터베이스 폴더는 "cubdl\datasets\data\picmus\database" 내에 있어야 합니다.
database_path = os.path.join("cubdl", "datasets", "data", "picmus", "database")

# 설정: simulation, contrast_speckle, iq
acq = "simulation"  # simulation 폴더 사용
target = "contrast_speckle"
dtype = "rf"

# PICMUSData 클래스가 내부적으로 아래와 같은 경로를 구성한다고 가정합니다.
# {database_path}/{acq}/{target}/{target}_{acqui}_dataset_{dtype}.hdf5
# simulation의 경우 acqui는 내부적으로 "simu"로 처리됩니다.
P = PICMUSData(database_path, acq, target, dtype)

# Define pixel grid limits (assume y == 0)
xlims = [P.ele_pos[0, 0], P.ele_pos[-1, 0]]
zlims = [5e-3, 55e-3]
wvln = P.c / P.fc
dx = wvln / 3
dz = dx  # Use square pixels
grid = make_pixel_grid(xlims, zlims, dx, dz)
fnum = 1

# Create a DAS_PW neural network for all angles, for 1 angle
dasN = DAS_PW(P, grid)
idx = len(P.angles) // 2  # Choose center angle for 1-angle DAS
das1 = DAS_PW(P, grid, idx)

# Stack the I and Q data in the innermost dimension
with tf.device("/gpu:0"):
    iqdata = np.stack((P.idata, P.qdata), axis=-1)

# Make 75-angle image
idasN, qdasN = dasN(iqdata)
idasN, qdasN = np.array(idasN), np.array(qdasN)
iqN = idasN + 1j * qdasN  # Combine I and Q data
bimgN = 20 * np.log10(np.abs(iqN))  # Log-compress
bimgN -= np.amax(bimgN)  # Normalize by maximum value

# Make 1-angle image
idas1, qdas1 = das1(iqdata)
idas1, qdas1 = np.array(idas1), np.array(qdas1)
iq1 = idas1 + 1j * qdas1  # Combine I and Q data
bimg1 = 20 * np.log10(np.abs(iq1))  # Log-compress
bimg1 -= np.amax(bimg1)  # Normalize by maximum value

# Display images via matplotlib
extent = [xlims[0] * 1e3, xlims[1] * 1e3, zlims[1] * 1e3, zlims[0] * 1e3]
plt.subplot(121)
plt.imshow(bimgN, vmin=-60, cmap="gray", extent=extent, origin="upper")
plt.title("%d angles" % len(P.angles))
plt.subplot(122)
plt.imshow(bimg1, vmin=-60, cmap="gray", extent=extent, origin="upper")
plt.title("Angle %d: %ddeg" % (idx, P.angles[idx] * 180 / np.pi))
plt.show()
