# File: FTDataLoaders.py
import os
import numpy as np
import h5py
from scipy.signal import hilbert
from cubdl.FocusedTxData import FocusedTxData


class OSLData(FocusedTxData):
    """University of Oslo 데이터를 로드하는 클래스."""

    def __init__(self, database_path, acq):
        pattern = f"OSL{acq:03d}"
        fname = []
        for root, dirs, files in os.walk(database_path):
            for file in files:
                if file.startswith(pattern) and file.endswith(".hdf5"):
                    fname.append(os.path.join(root, file))
        if not fname:
            raise FileNotFoundError(
                f"File not found. Pattern: {pattern}*.hdf5 in {database_path}"
            )
        # 1과 15도 허용
        assert acq in [
            1,
            8,
            9,
            11,
            12,
            13,
            14,
            15,
        ], "Plane Wave Data. Use PWDataLoaders"

        with h5py.File(fname[0], "r") as f:
            # Raw data load
            self.idata = np.array(f["channel_data"], dtype="float32")
            self.qdata = np.imag(hilbert(self.idata, axis=-1))
            self.tx_ori = np.array(f["transmit_origin"], dtype="float32").T
            self.tx_dir = np.array(f["transmit_direction"], dtype="float32").T
            self.tx_foc = np.array(f["transmit_focus"], dtype="float32")
            self.fc = np.array(f["modulation_frequency"]).item()
            self.fs = np.array(f["sampling_frequency"]).item()
            self.c = np.array(f["sound_speed"]).item()

            # time_zero: per-transmit array
            raw_t0 = np.array(f["start_time"], dtype="float32").flatten()
            nxmits = self.idata.shape[0]
            if raw_t0.size == 1:
                t0 = raw_t0.item() * np.ones((nxmits,), dtype="float32")
            else:
                t0 = raw_t0
            # acquisition 별 조정
            if acq in [8, 9]:
                self.tx_dir *= 0
                t0 = -np.min(raw_t0) * np.ones((nxmits,), dtype="float32")
            else:
                self.tx_ori *= 0
                t0 = -t0
            self.time_zero = t0

            # 기타 메타데이터
            self.fdemod = 0
            self.ele_pos = np.array(f["element_positions"], dtype="float32").T
            self.ele_pos[:, 0] -= np.mean(self.ele_pos[:, 0])
            # tx_foc array
            self.tx_foc = self.tx_foc[0, 0] * np.ones((nxmits,), dtype="float32")

        super().validate()


class JHUData(FocusedTxData):
    """Johns Hopkins University 데이터를 로드하는 클래스."""

    def __init__(self, database_path, acq):
        pattern = f"JHU{acq:03d}"
        fname = []
        for root, dirs, files in os.walk(database_path):
            for file in files:
                if file.startswith(pattern) and file.endswith(".hdf5"):
                    fname.append(os.path.join(root, file))
        if not fname:
            raise FileNotFoundError(
                f"File not found. Pattern: {pattern}.hdf5 in {database_path}"
            )
        assert acq in [1, 2, 19, 20, 21, 22, 23], "Plane Wave Data. Use PWDataLoaders"

        with h5py.File(fname[0], "r") as f:
            if acq <= 2:
                # Phantom data load
                self.idata = np.array(f["channel_data"], dtype="float32")
                self.qdata = np.imag(hilbert(self.idata, axis=-1))
                self.tx_foc = np.array(f["transmit_focus"], dtype="float32")
                self.fc = np.array(f["modulation_frequency"]).item()
                self.fs = np.array(f["sampling_frequency"]).item()
                self.fdemod = 0
                self.ele_pos = np.zeros((128, 3), dtype="float32")
                self.ele_pos[:, 0] = np.arange(128) * f["pitch"]
                self.ele_pos[:, 0] -= np.mean(self.ele_pos[:, 0])
                self.c = 1540.0
                self.tx_ori = np.zeros((256, 3), dtype="float32")
                self.tx_ori[:, 0] = np.arange(256) * f["pitch"] / 2
                self.tx_ori[:, 0] -= np.mean(self.tx_ori[:, 0])
                self.time_zero = np.zeros((256,), dtype="float32")
                nxmits = self.idata.shape[0]
                self.tx_foc = self.tx_foc[0] * np.ones((nxmits,), dtype="float32")
                self.tx_dir = np.zeros((256, 2), dtype="float32")
            else:
                # In vivo data load
                self.idata = np.array(f["channel_data"], dtype="float32")
                self.qdata = np.imag(hilbert(self.idata, axis=-1))
                self.tx_ori = np.array(f["transmit_origin"], dtype="float32").T / 2
                self.tx_dir = np.array(f["transmit_direction"], dtype="float32").T
                self.tx_foc = np.array(f["transmit_focus"], dtype="float32")
                self.fc = np.array(f["modulation_frequency"]).item()
                self.fs = np.array(f["sampling_frequency"]).item()
                self.time_zero = np.array(f["start_time"], dtype="float32")[0]
                self.fdemod = 0
                self.ele_pos = np.array(f["element_positions"], dtype="float32").T
                self.ele_pos[:, 0] -= np.mean(self.ele_pos[:, 0])
                self.c = 1540.0

        super().validate()


def get_filelist(data_type="all"):
    if data_type == "phantom":
        filelist = {"JHU": [19, 20, 21, 22, 23]}
    elif data_type == "invivo":
        filelist = {"OSL": [1, 8, 9, 11, 12, 13, 14, 15], "JHU": [1, 2]}
    elif data_type == "all":
        filelist = {
            "OSL": [1, 8, 9, 11, 12, 13, 14, 15],
            "JHU": [1, 2, 19, 20, 21, 22, 23],
        }
    else:
        filelist = {"OSL": [8, 9], "JHU": [1, 2]}
    return filelist


# import os
# import numpy as np
# import h5py
# from scipy.signal import hilbert
# from cubdl.FocusedTxData import FocusedTxData


# class OSLData(FocusedTxData):
#     """University of Oslo 데이터를 로드하는 클래스."""

#     def __init__(self, database_path, acq):
#         # 파일 이름 패턴 구성 (예: "OSL008")
#         pattern = "OSL{:03d}".format(acq)
#         fname = []
#         # 지정된 데이터베이스 경로 내의 모든 파일 검색
#         for root, dirs, files in os.walk(database_path):
#             for file in files:
#                 if file.startswith(pattern) and file.endswith(".hdf5"):
#                     fname.append(os.path.join(root, file))
#         if not fname:
#             raise FileNotFoundError(
#                 f"File not found. Pattern: {pattern}*.hdf5 in {database_path}. "
#                 "경로 문자열은 raw string (r'경로') 또는 역슬래시(\\) 두 개로 입력하세요."
#             )
#         # PWDataLoaders.py와 동일한 acq 번호 제한
#         assert acq in [8, 9, 11, 12, 13, 14], "Plane Wave Data. Use PWDataLoaders"

#         # 파일 열어서 데이터 로드
#         with h5py.File(fname[0], "r") as f:
#             self.idata = np.array(f["channel_data"], dtype="float32")
#             self.qdata = np.imag(hilbert(self.idata, axis=-1))
#             self.tx_ori = np.array(f["transmit_origin"], dtype="float32").T
#             self.tx_dir = np.array(f["transmit_direction"], dtype="float32").T
#             self.tx_foc = np.array(f["transmit_focus"], dtype="float32")
#             self.fc = np.array(f["modulation_frequency"]).item()
#             self.fs = np.array(f["sampling_frequency"]).item()
#             self.c = np.array(f["sound_speed"]).item()
#             self.time_zero = np.array(f["start_time"], dtype="float32")[0]
#             self.fdemod = 0
#             self.ele_pos = np.array(f["element_positions"], dtype="float32").T
#             self.ele_pos[:, 0] -= np.mean(self.ele_pos[:, 0])
#             nxmits = self.idata.shape[0]
#             self.tx_foc = self.tx_foc[0, 0] * np.ones((nxmits,), dtype="float32")
#             if acq in [8, 9]:
#                 self.tx_dir *= 0
#                 self.time_zero = 0 * self.time_zero - np.min(self.time_zero)
#             else:
#                 self.tx_ori *= 0
#                 self.time_zero = -1 * self.time_zero
#         super().validate()


# class JHUData(FocusedTxData):
#     """Johns Hopkins University 데이터를 로드하는 클래스."""

#     def __init__(self, database_path, acq):
#         # 파일 이름 패턴 구성 (예: "JHU001")
#         pattern = "JHU{:03d}".format(acq)
#         fname = []
#         for root, dirs, files in os.walk(database_path):
#             for file in files:
#                 if file.startswith(pattern) and file.endswith(".hdf5"):
#                     fname.append(os.path.join(root, file))
#         if not fname:
#             raise FileNotFoundError(
#                 f"File not found. Pattern: {pattern}.hdf5 in {database_path}. "
#                 "경로 문자열은 raw string (r'경로') 또는 역슬래시(\\) 두 개로 입력하세요."
#             )
#         # PWDataLoaders.py와 동일 acq 번호 제한
#         assert acq in [1, 2, 19, 20, 21, 22, 23], "Plane Wave Data. Use PWDataLoaders"

#         with h5py.File(fname[0], "r") as f:
#             if acq <= 2:
#                 self.idata = np.array(f["channel_data"], dtype="float32")
#                 self.qdata = np.imag(hilbert(self.idata, axis=-1))
#                 self.tx_foc = np.array(f["transmit_focus"], dtype="float32")
#                 self.fc = np.array(f["modulation_frequency"]).item()
#                 self.fs = np.array(f["sampling_frequency"]).item()
#                 self.fdemod = 0
#                 self.ele_pos = np.zeros((128, 3), dtype="float32")
#                 self.ele_pos[:, 0] = np.arange(128) * f["pitch"]
#                 self.ele_pos[:, 0] -= np.mean(self.ele_pos[:, 0])
#                 self.c = 1540.0
#                 self.tx_ori = np.zeros((256, 3), dtype="float32")
#                 self.tx_ori[:, 0] = np.arange(256) * f["pitch"] / 2
#                 self.tx_ori[:, 0] -= np.mean(self.tx_ori[:, 0])
#                 self.time_zero = np.zeros((256,), dtype="float32")
#                 nxmits = self.idata.shape[0]
#                 self.tx_foc = self.tx_foc[0] * np.ones((nxmits,), dtype="float32")
#                 self.tx_dir = np.zeros((256, 2), dtype="float32")
#             else:
#                 self.idata = np.array(f["channel_data"], dtype="float32")
#                 self.qdata = np.imag(hilbert(self.idata, axis=-1))
#                 self.tx_ori = np.array(f["transmit_origin"], dtype="float32").T
#                 self.tx_ori = self.tx_ori / 2
#                 self.tx_dir = np.array(f["transmit_direction"], dtype="float32").T
#                 self.tx_foc = np.array(f["transmit_focus"], dtype="float32")
#                 self.fc = np.array(f["modulation_frequency"]).item()
#                 self.fs = np.array(f["sampling_frequency"]).item()
#                 self.time_zero = np.array(f["start_time"], dtype="float32")[0]
#                 self.fdemod = 0
#                 self.ele_pos = np.array(f["element_positions"], dtype="float32").T
#                 self.ele_pos[:, 0] -= np.mean(self.ele_pos[:, 0])
#                 self.c = 1540.0
#         super().validate()


# def get_filelist(data_type="all"):
#     if data_type == "phantom":
#         filelist = {"JHU": [19, 20, 21, 22, 23]}
#     elif data_type == "invivo":
#         filelist = {"OSL": [8, 9, 11, 12, 13, 14], "JHU": [1, 2]}
#     elif data_type == "all":
#         filelist = {"OSL": [8, 9, 11, 12, 13, 14], "JHU": [1, 2, 19, 20, 21, 22, 23]}
#     else:
#         filelist = {"OSL": [8, 9], "JHU": [1, 2]}
#     return filelist
