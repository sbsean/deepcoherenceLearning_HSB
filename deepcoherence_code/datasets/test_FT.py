# File:       test_FT.py
# Author:     Dongwoon Hyun (dongwoon.hyun@stanford.edu)
# Created on: 2020-03-12
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from cubdl.das_torch import DAS_FT
from datasets.FTDataLoaders import OSLData, JHUData, get_filelist
from cubdl.PixelGrid import make_foctx_grid, make_pixel_grid
from scipy.interpolate import griddata

# 위 모듈들은 파일 경로 처리, GPU 기반 텐서 연산, 이미지 시각화, 수치 계산,
# DAS 빔포밍, 데이터 로더, 픽셀 그리드 생성, 그리고 그리드 보간을 위해 임포트됨

# 현재 스크립트 파일의 디렉토리를 기준으로 base_dir 변수 설정
base_dir = os.path.dirname(os.path.abspath(__file__))

# 'datasets' 폴더가 중복되지 않도록 경로를 구성
# 만약 test_FT.py가 이미 "cubdl/datasets" 폴더 내에 있다면,
# "data/cubdl_data/3_Additional_CUBDL_Data/Focused_Data"만 추가됨
database_path = os.path.join(
    base_dir, "data", "cubdl_data", "3_Additional_CUBDL_Data", "Focused_Data"
)

# Focused Data 파일 목록을 얻기 위해 get_filelist 함수 호출
# data_type 매개변수는 "all", "phantom", "invivo" 중 선택 가능
filelist = get_filelist(data_type="all")

# filelist에 있는 각 데이터 소스(예: "OSL", "JHU")에 대해 반복
for data_source in filelist:
    # 각 데이터 소스 내의 acq 번호(데이터셋 번호)별로 반복
    for acq in filelist[data_source]:
        if data_source == "OSL":
            # OSL 데이터 소스의 경우, OSLData 클래스를 이용하여 데이터 로드
            F = OSLData(database_path, acq)
            # acq 번호에 따라 최대 범위(rmax)와 스캔 컨버전(scan_convert) 여부 결정
            rmax = 38e-3 if (acq == 8 or acq == 9) else 125e-3
            scan_convert = False if (acq == 8 or acq == 9) else True
            drange = 60  # 이미지 다이내믹 레인지 (dB)
        elif data_source == "JHU":
            # JHU 데이터 소스의 경우, JHUData 클래스를 이용하여 데이터 로드
            F = JHUData(database_path, acq)
            if acq <= 2:
                rmax = 40e-3
                scan_convert = False
            else:
                rmax = 60e-3
                scan_convert = True
            drange = 50  # 이미지 다이내믹 레인지 (dB)
        else:
            # 지원하지 않는 데이터 소스가 입력된 경우 예외 발생
            raise NotImplementedError

        # 픽셀 그리드의 제한을 정의 (y 좌표는 0으로 가정)
        wvln = F.c / F.fc  # 파장 계산 (음속 / 변조 주파수)
        dr = wvln / 4  # 그리드 간격을 파장의 1/4로 설정
        rlims = [0, rmax]  # 반경 한계: 0부터 rmax까지
        grid = make_foctx_grid(rlims, dr, F.tx_ori, F.tx_dir)  # 포커스 전송 그리드 생성
        fnum = 0  # 전송 번호 (default 값)

        # 데이터를 GPU 텐서로 변환 (float 타입, cuda:0 디바이스 사용)
        idata = torch.tensor(F.idata, dtype=torch.float, device=torch.device("cuda:0"))
        qdata = torch.tensor(F.qdata, dtype=torch.float, device=torch.device("cuda:0"))
        x = (idata, qdata)  # 실수부와 허수부 데이터를 튜플로 묶음

        # DAS_FT 객체 생성하여 DAS 빔포밍 수행
        das = DAS_FT(F, grid, rxfnum=fnum)
        idas, qdas = das(x)  # DAS 빔포밍 결과 (실수부, 허수부)
        # 결과를 GPU에서 CPU로 이동 후 numpy 배열로 변환
        idas, qdas = idas.detach().cpu().numpy(), qdas.detach().cpu().numpy()
        iq = idas + 1j * qdas  # 복소수 형태의 IQ 데이터 생성
        bimg = np.abs(iq).T  # 절대값(진폭) 취한 후 전치하여 이미지 데이터 생성

        # 스캔 컨버전(Scan Conversion)이 필요한 경우 수행
        if scan_convert:
            # x축 및 z축의 한계를 정의 (x: 좌우, z: 깊이)
            xlims = rlims[1] * np.array([-0.7, 0.7])
            zlims = rlims[1] * np.array([0, 1])
            # 픽셀 그리드 생성 (해상도는 파장의 절반)
            img_grid = make_pixel_grid(xlims, zlims, wvln / 2, wvln / 2)
            # 기존 그리드의 축 순서를 전치 (swap axes)
            grid = np.transpose(grid, (1, 0, 2))
            # 보간을 위한 원본 그리드 포인트 생성: (z, x) 순서
            g1 = np.stack((grid[:, :, 2], grid[:, :, 0]), -1).reshape(-1, 2)
            # 보간 대상 그리드 포인트 생성: (z, x) 순서
            g2 = np.stack((img_grid[:, :, 2], img_grid[:, :, 0]), -1).reshape(-1, 2)
            # griddata를 사용하여 선형 보간 수행, 작은 값은 1e-10으로 대체
            bsc = griddata(g1, bimg.reshape(-1), g2, "linear", 1e-10)
            # 보간 결과를 원래 이미지 그리드 형태로 재배열
            bimg = np.reshape(bsc, img_grid.shape[:2])
            # 그리드를 새로운 이미지 그리드로 업데이트 (축 전치)
            grid = img_grid.transpose(1, 0, 2)

        bimg = 20 * np.log10(bimg)  # 진폭 데이터를 dB 단위로 로그 압축
        bimg -= np.amax(bimg)  # 최대값으로 정규화 (최대가 0 dB가 되도록)

        # matplotlib을 사용하여 이미지를 표시
        extent = [grid[0, 0, 0], grid[-1, 0, 0], grid[0, -1, 2], grid[0, 0, 2]]
        extent = np.array(extent) * 1e3  # 좌표를 mm 단위로 변환
        plt.clf()  # 현재 figure 초기화
        plt.imshow(bimg, vmin=-drange, cmap="gray", extent=extent, origin="upper")
        plt.title(
            "%s%03d (c = %d m/s)" % (data_source, acq, np.round(F.c))
        )  # 제목 설정
        plt.colorbar()  # 컬러바 표시
        plt.pause(0.01)  # 잠시 대기 (애니메이션 효과를 위해)
        plt.show()  # 이미지 출력

        # 저장할 디렉토리 존재 여부 확인, 없으면 생성
        savedir = os.path.join(base_dir, "data", "test_ft_images", data_source)
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        # 결과 이미지를 JPEG 파일로 저장
        plt.savefig(os.path.join(savedir, "%s%03d.jpg" % (data_source, acq)))
