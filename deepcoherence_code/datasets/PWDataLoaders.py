# PWDataLoaders.py
import random
import os
import h5py
import numpy as np
from glob import glob
from scipy.signal import hilbert, convolve
from cubdl.PlaneWaveData import PlaneWaveData
from cubdl.PlaneWaveData import PICMUSData, InVivoPICMUSData

def load_data(data_source, acq):
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 기존 datadir1, datadir3 정의
    datadir1 = os.path.join(base_dir, "data", "cubdl_data", "1_CUBDL_Task1_Data")
    datadir3 = os.path.join(
        base_dir, "data", "cubdl_data", "3_Additional_CUBDL_Data", "Plane_Wave_Data"
    )

    # 추가: JHU 데이터 전용 디렉터리
    datadir_jhu = os.path.join(
        base_dir, "data", "cubdl_data", "2_Post_CUBDL_JHU_Breast_Data"
    )

    if data_source == "MYO":
        if acq in [1, 2, 3, 4, 5, 6]:
            database_path = datadir1
        else:
            raise ValueError("MYO%03d is not a valid plane wave acquisition." % acq)
        P = MYOData(database_path, acq)
        xlims = [P.ele_pos[0, 0], P.ele_pos[-1, 0]]
        zlims = [5e-3, 55e-3]

    elif data_source == "UFL":
        if acq in [1, 2, 4, 5]:
            database_path = datadir1
        elif acq == 3:
            database_path = datadir3
        else:
            raise ValueError("UFL%03d is not a valid plane wave acquisition." % acq)
        P = UFLData(database_path, acq)
        xlims = [P.ele_pos[0, 0], P.ele_pos[-1, 0]]
        zlims = [10e-3, 50e-3]

    elif data_source == "EUT":
        if acq in [3, 6]:
            database_path = datadir1
        elif acq in [1, 2, 4, 5]:
            database_path = datadir3
        else:
            raise ValueError("EUT%03d is not a valid plane wave acquisition." % acq)
        P = EUTData(database_path, acq)
        xlims = [P.ele_pos[0, 0], P.ele_pos[-1, 0]]
        zlims = [10e-3, 80e-3]

    elif data_source == "INS":
        if acq in list(range(1, 27)):
            database_path = datadir1
        # elif 1 <= acq <= 26:
        #     database_path = datadir3
        else:
            raise ValueError("INS%03d is not a valid plane wave acquisition." % acq)
        P = INSData(database_path, acq)
        xlims = [P.ele_pos[0, 0], P.ele_pos[-1, 0]]
        zlims = [10e-3, 60e-3]
        if acq >= 13:
            zlims = [10e-3, 50e-3]

    elif data_source == "OSL":
        if acq in [2, 3, 4, 5, 6]:
            database_path = datadir3
        elif acq in [7]:
            database_path = datadir1
        elif acq in [10]:
            database_path = os.path.join(datadir1, "OSL010")
        else:
            raise ValueError("OSL%03d is not a valid plane wave acquisition." % acq)
        P = OSLData(database_path, acq)
        xlims = [P.ele_pos[0, 0], P.ele_pos[-1, 0]]
        zlims = [10e-3, 65e-3]
        if acq == 10:
            zlims = [5e-3, 50e-3]

    elif data_source == "TSH":
        if acq in [2]:
            database_path = os.path.join(datadir1, "TSH002")
        elif 3 <= acq <= 501:
            database_path = os.path.join(datadir3, "TSH")
        else:
            raise ValueError("TSH%03d is not a valid plane wave acquisition." % acq)
        P = TSHData(database_path, acq)
        xlims = [P.ele_pos[0, 0], P.ele_pos[-1, 0]]
        zlims = [10e-3, 45e-3]

    elif data_source == "JHU":
        database_path = datadir_jhu
        P = JHUData(database_path, acq)
        xlims = [P.ele_pos[0, 0], P.ele_pos[-1, 0]]
        zlims = [0e-3, 30e-3]

    elif data_source == "PICMUS":
        if acq == 1:
            database_path = os.path.join(
                base_dir,
                "data",
                "picmus",
                "database",
                "experiments",
                "contrast_speckle",
            )
            P = PICMUSData(
                os.path.join(base_dir, "data", "picmus", "database"),
                "experiments",
                "contrast_speckle",
                "iq",
            )
            zlims = [0e-3, 42e-3]

        elif acq == 2:
            database_path = os.path.join(
                base_dir,
                "data",
                "picmus",
                "database",
                "experiments",
                "resolution_distorsion",
            )
            P = PICMUSData(
                os.path.join(base_dir, "data", "picmus", "database"),
                "experiments",
                "resolution_distorsion",
                "iq",
            )
            zlims = [0e-3, 42e-3]

        elif acq == 3:
            database_path = os.path.join(
                base_dir, "data", "picmus", "database", "simulation", "contrast_speckle"
            )
            P = PICMUSData(
                os.path.join(base_dir, "data", "picmus", "database"),
                "simulation",
                "contrast_speckle",
                "iq",
            )
            zlims = [5e-3, 50e-3]

        elif acq == 4:
            database_path = os.path.join(
                base_dir,
                "data",
                "picmus",
                "database",
                "simulation",
                "resolution_distorsion",
            )
            P = PICMUSData(
                os.path.join(base_dir, "data", "picmus", "database"),
                "simulation",
                "resolution_distorsion",
                "iq",
            )
            zlims = [5e-3, 50e-3]

        elif acq == 5:
            filepath = os.path.join(
                base_dir,
                "data",
                "picmus",
                "database",
                "in_vivo",
                "carotid_cross",
                "carotid_cross_expe_dataset_iq.hdf5",
            )
            P = InVivoPICMUSData(filepath=filepath, dtype="iq")
            zlims = [0e-3, 55e-3]

        elif acq == 6:
            filepath = os.path.join(
                base_dir,
                "data",
                "picmus",
                "database",
                "in_vivo",
                "carotid_long",
                "carotid_long_expe_dataset_iq.hdf5",
            )
            P = InVivoPICMUSData(filepath=filepath, dtype="iq")
            zlims = [0e-3, 55e-3]

        else:
            raise ValueError("PICMUS acquisition number must be 1–6.")

        xlims = [P.ele_pos[0, 0], P.ele_pos[-1, 0]]

    else:
        raise NotImplementedError

    return P, xlims, zlims


# PICMUS 데이터 로더 클래스 (IQ 데이터로 로드)
class PICMUSData(PlaneWaveData):
    """Load IQ data from PICMUS dataset.
    파일 구조:
      - 상위 그룹 "US"
        - 그룹 "US_DATASET0000" 내에:
          - angles: (75,) float32
          - data: Group
              - real: (75, 128, X) float32
              - imag: (75, 128, X) float32
          - modulation_frequency: (1,) float32
          - sampling_frequency: (1,) float32
          - sound_speed: (1,) float32
          - probe_geometry: (3, 128) float32
    """

    def __init__(self, database_path, acq):
        moniker = "PICMUS{:03d}.hdf5".format(acq)
        fname = [
            y for x in os.walk(database_path) for y in glob(os.path.join(x[0], moniker))
        ]
        assert fname, "File not found for PICMUS acquisition %03d." % acq
        f = h5py.File(fname[0], "r")
        us_grp = f["US"]["US_DATASET0000"]
        self.angles = np.array(us_grp["angles"])
        real = np.array(us_grp["data"]["real"], dtype="float32")
        imag = np.array(us_grp["data"]["imag"], dtype="float32")
        iqdata = real + 1j * imag
        self.idata = np.real(iqdata)
        self.qdata = np.imag(iqdata)
        self.fc = np.array(us_grp["modulation_frequency"]).item()
        self.fs = np.array(us_grp["sampling_frequency"]).item()
        self.c = np.array(us_grp["sound_speed"]).item()
        probe_geom = np.array(us_grp["probe_geometry"], dtype="float32")
        self.ele_pos = probe_geom.T
        self.ele_pos[:, 0] -= np.mean(self.ele_pos[:, 0])
        self.time_zero = np.zeros(len(self.angles), dtype="float32")
        for i, a in enumerate(self.angles):
            self.time_zero[i] = self.ele_pos[-1, 0] * np.abs(np.sin(a)) / self.c
        self.fdemod = 0
        super().validate()


# ----------------------------------------------------------------------------
# TSHData: 수정된 버전 – 각 angle별로 Hilbert 변환을 수행하여 메모리 사용량을 낮춤
class TSHData(PlaneWaveData):
    """Load data from Tsinghua University."""

    def __init__(self, database_path, acq):
        from scipy.signal import hilbert

        moniker = "TSH{:03d}".format(acq) + "*.hdf5"
        fname = [
            y for x in os.walk(database_path) for y in glob(os.path.join(x[0], moniker))
        ]
        assert fname, "File not found."
        f = h5py.File(fname[0], "r")
        self.angles = np.array(f["angles"])
        # HDF5 데이터셋에서 불필요한 복사를 피하기 위해 [...] 사용 후 자료형 변환
        self.idata = f["channel_data"][...].astype("float32")
        # idata의 shape 재구성: (128, len(angles), X)로 만들고 전치 -> (len(angles), 128, X)
        self.idata = np.reshape(self.idata, (128, len(self.angles), -1))
        self.idata = np.transpose(self.idata, (1, 0, 2))
        # 각 angle별로 Hilbert 변환 수행 (메모리 사용량 감소)
        qdata_list = []
        for i in range(self.idata.shape[0]):
            qdata_list.append(np.imag(hilbert(self.idata[i], axis=-1)))
        self.qdata = np.stack(qdata_list, axis=0)
        self.fc = np.array(f["modulation_frequency"]).item()
        self.fs = np.array(f["sampling_frequency"]).item()
        self.c = 1540
        self.time_zero = np.zeros((len(self.angles),), dtype="float32")
        self.fdemod = 0
        pitch = 0.3e-3
        nelems = self.idata.shape[1]
        xpos = np.arange(nelems) * pitch
        xpos -= np.mean(xpos)
        self.ele_pos = np.stack(
            [xpos, np.zeros_like(xpos), np.zeros_like(xpos)], axis=1
        )
        for i, a in enumerate(self.angles):
            self.time_zero[i] = self.ele_pos[-1, 0] * np.abs(np.sin(a)) / self.c
        super().validate()


class MYOData(PlaneWaveData):
    """Load data from Mayo Clinic."""

    def __init__(self, database_path, acq):
        moniker = "MYO{:03d}".format(acq) + "*.hdf5"
        fname = [
            y for x in os.walk(database_path) for y in glob(os.path.join(x[0], moniker))
        ]
        assert fname, "File not found."
        f = h5py.File(fname[0], "r")
        if acq == 1:
            sound_speed = 1580
        elif acq == 2:
            sound_speed = 1583
        elif acq == 3:
            sound_speed = 1578
        elif acq == 4:
            sound_speed = 1572
        elif acq == 5:
            sound_speed = 1562
        else:
            sound_speed = 1581
        self.idata = np.array(f["channel_data"], dtype="float32")
        self.qdata = np.imag(hilbert(self.idata, axis=-1))
        self.angles = np.array(f["angles"])
        self.fc = np.array(f["modulation_frequency"]).item()
        self.fs = np.array(f["sampling_frequency"]).item()
        self.c = sound_speed
        self.time_zero = np.zeros((len(self.angles),), dtype="float32")
        self.fdemod = 0
        pitch = 0.3e-3
        nelems = self.idata.shape[1]
        xpos = np.arange(nelems) * pitch
        xpos -= np.mean(xpos)
        self.ele_pos = np.stack(
            [xpos, np.zeros_like(xpos), np.zeros_like(xpos)], axis=1
        )
        for i, a in enumerate(self.angles):
            self.time_zero[i] = self.ele_pos[-1, 0] * np.abs(np.sin(a)) / self.c
        super().validate()


class UFLData(PlaneWaveData):
    """Load data from UNIFI."""

    def __init__(self, database_path, acq):
        moniker = "UFL{:03d}".format(acq) + "*.hdf5"
        fname = [
            y for x in os.walk(database_path) for y in glob(os.path.join(x[0], moniker))
        ]
        assert fname, "File not found."
        f = h5py.File(fname[0], "r")
        if acq == 1:
            sound_speed = 1526
        elif acq in [2, 4, 5]:
            sound_speed = 1523
        else:
            sound_speed = 1525
        self.idata = np.array(f["channel_data"], dtype="float32")
        self.qdata = np.imag(hilbert(self.idata, axis=-1))
        self.angles = np.array(f["angles"]) * np.pi / 180
        self.fc = np.array(f["modulation_frequency"]).item()
        self.fs = np.array(f["channel_data_sampling_frequency"]).item()
        self.c = sound_speed
        self.time_zero = -1 * np.array(f["channel_data_t0"], dtype="float32")
        self.fdemod = self.fc
        pitch = 0.245e-3
        nelems = self.idata.shape[1]
        xpos = np.arange(nelems) * pitch
        xpos -= np.mean(xpos)
        self.ele_pos = np.stack(
            [xpos, np.zeros_like(xpos), np.zeros_like(xpos)], axis=1
        )
        if self.time_zero.size == 1:
            self.time_zero = np.ones_like(self.angles) * self.time_zero
        data = self.idata + 1j * self.qdata
        phase = np.reshape(np.arange(self.idata.shape[2], dtype="float"), (1, 1, -1))
        phase *= self.fdemod / self.fs
        data *= np.exp(-2j * np.pi * phase)
        dsfactor = int(np.floor(self.fs / self.fc))
        kernel = np.ones((1, 1, dsfactor), dtype="float") / dsfactor
        data = convolve(data, kernel, "same")
        data = data[:, :, ::dsfactor]
        self.fs /= dsfactor
        self.idata = np.real(data)
        self.qdata = np.imag(data)
        super().validate()


class EUTData(PlaneWaveData):
    """Load data from TU/e."""

    def __init__(self, database_path, acq):
        moniker = "EUT{:03d}".format(acq) + "*.hdf5"
        fname = [
            y for x in os.walk(database_path) for y in glob(os.path.join(x[0], moniker))
        ]
        assert fname, "File not found."
        f = h5py.File(fname[0], "r")
        if acq == 1:
            sound_speed = 1603
        elif acq == 2:
            sound_speed = 1618
        elif acq == 3:
            sound_speed = 1607
        elif acq == 4:
            sound_speed = 1614
        elif acq == 5:
            sound_speed = 1495
        else:
            sound_speed = 1479
        self.idata = np.array(f["channel_data"], dtype="float32")
        self.qdata = np.imag(hilbert(self.idata, axis=-1))
        self.angles = np.array(f["transmit_direction"])[:, 0]
        self.fc = np.array(f["modulation_frequency"]).item()
        self.fs = np.array(f["sampling_frequency"]).item()
        self.c = sound_speed
        self.time_zero = np.array(f["start_time"], dtype="float32")[0]
        self.fdemod = 0
        self.ele_pos = np.array(f["element_positions"], dtype="float32").T
        self.ele_pos[:, 0] -= np.mean(self.ele_pos[:, 0])
        for i, a in enumerate(self.angles):
            self.time_zero[i] = self.ele_pos[-1, 0] * np.abs(np.sin(a)) / self.c
        self.time_zero += 10 / self.fc
        super().validate()


class INSData(PlaneWaveData):
    """Load data from INSERM."""

    def __init__(self, database_path, acq):
        moniker = "INS{:03d}".format(acq) + "*.hdf5"
        fname = [
            y for x in os.walk(database_path) for y in glob(os.path.join(x[0], moniker))
        ]
        assert fname, "File not found."
        f = h5py.File(fname[0], "r")
        if acq == 1:
            sound_speed = 1521
        elif acq == 2:
            sound_speed = 1517
        elif acq == 3:
            sound_speed = 1506
        elif acq == 4:
            sound_speed = 1501
        elif acq == 5:
            sound_speed = 1506
        elif acq == 6:
            sound_speed = 1509
        elif acq == 7:
            sound_speed = 1490
        elif acq == 8:
            sound_speed = 1504
        elif acq == 9:
            sound_speed = 1473
        elif acq == 10:
            sound_speed = 1502
        elif acq == 11:
            sound_speed = 1511
        elif acq == 12:
            sound_speed = 1535
        elif acq == 13:
            sound_speed = 1453
        elif acq == 14:
            sound_speed = 1542
        elif acq == 15:
            sound_speed = 1539
        elif acq == 16:
            sound_speed = 1466
        elif acq == 17:
            sound_speed = 1462
        elif acq == 18:
            sound_speed = 1479
        elif acq == 19:
            sound_speed = 1469
        elif acq == 20:
            sound_speed = 1464
        elif acq == 21:
            sound_speed = 1508
        elif acq == 22:
            sound_speed = 1558
        elif acq == 23:
            sound_speed = 1463
        elif acq == 24:
            sound_speed = 1547
        elif acq == 25:
            sound_speed = 1477
        elif acq == 26:
            sound_speed = 1497
        else:
            sound_speed = 1540
        self.idata = np.array(f["channel_data"], dtype="float32")
        self.qdata = np.imag(hilbert(self.idata, axis=-1))
        self.angles = np.linspace(-16, 16, self.idata.shape[0]) * np.pi / 180
        self.fc = np.array(f["modulation_frequency"]).item()
        self.fs = np.array(f["sampling_frequency"]).item()
        self.c = sound_speed
        self.time_zero = -1 * np.array(f["start_time"], dtype="float32")[0]
        self.fdemod = 0
        self.ele_pos = np.array(f["element_positions"], dtype="float32").T
        self.ele_pos[:, 0] -= np.mean(self.ele_pos[:, 0])
        for i, a in enumerate(self.angles):
            self.time_zero[i] += self.ele_pos[-1, 0] * np.abs(np.sin(a)) / self.c
        super().validate()


class OSLData(PlaneWaveData):
    """Load data from University of Oslo."""

    def __init__(self, database_path, acq):
        moniker = "OSL{:03d}".format(acq) + ".hdf5"
        fname = [
            y for x in os.walk(database_path) for y in glob(os.path.join(x[0], moniker))
        ]
        assert fname, "File not found."
        assert acq in [2, 3, 4, 5, 6, 7, 10], "Focused Data. Use FTDataLoaders"
        f = h5py.File(fname[0], "r")
        if acq == 2:
            sound_speed = 1536
        elif acq == 3:
            sound_speed = 1543
        elif acq == 4:
            sound_speed = 1538
        elif acq == 5:
            sound_speed = 1539
        elif acq == 6:
            sound_speed = 1541
        elif acq == 7:
            sound_speed = 1540
        else:
            sound_speed = 1540
        self.idata = np.array(f["channel_data"], dtype="float32")
        self.qdata = np.imag(hilbert(self.idata, axis=-1))
        self.angles = np.array(f["transmit_direction"][0], dtype="float32")
        self.fc = np.array(f["modulation_frequency"]).item()
        self.fs = np.array(f["sampling_frequency"]).item()
        self.c = sound_speed
        self.time_zero = -1 * np.array(f["start_time"], dtype="float32")[0]
        self.fdemod = 0
        self.ele_pos = np.array(f["element_positions"], dtype="float32").T
        self.ele_pos[:, 0] -= np.mean(self.ele_pos[:, 0])
        super().validate()


class JHUData(PlaneWaveData):
    """Load data from Johns Hopkins University."""

    def __init__(self, database_path, acq):
        moniker = "JHU{:03d}".format(acq) + ".hdf5"
        fname = [
            y for x in os.walk(database_path) for y in glob(os.path.join(x[0], moniker))
        ]
        assert fname, "File not found."
        assert acq in list(range(24, 35)), "Focused Data. Use FTDataLoaders"
        f = h5py.File(fname[0], "r")
        self.idata = np.array(f["channel_data"], dtype="float32")
        self.qdata = np.imag(hilbert(self.idata, axis=-1))
        self.angles = np.array(f["angles"])
        self.fc = np.array(f["modulation_frequency"]).item()
        self.fs = np.array(f["sampling_frequency"]).item()
        self.c = np.array(f["sound_speed"]).item()
        self.time_zero = -1 * np.array(f["time_zero"], dtype="float32")
        self.fdemod = 0
        xpos = np.array(f["element_positions"], dtype="float32").T
        self.ele_pos = np.stack(
            [xpos, np.zeros_like(xpos), np.zeros_like(xpos)], axis=1
        )
        self.zlims = np.array([0e-3, self.idata.shape[2] * self.c / self.fs / 2])
        self.xlims = np.array([self.ele_pos[0, 0], self.ele_pos[-1, 0]])
        self.time_zero = np.zeros((len(self.angles),), dtype="float32")
        for i, a in enumerate(self.angles):
            self.time_zero[i] = self.ele_pos[-1, 0] * np.abs(np.sin(a)) / self.c

        self.time_zero -= 10 / self.fc #offset

        super().validate()


def get_filelist(data_type="task1"):
    if data_type == "all": # 6 + 2 + 6 + 4 + 2 + 26 + 11 + 500
        print("Warning: TSH has 500 files. This will take a while.")
        filelist = {
            "PICMUS": [0, 1, 2, 3, 4, 5],
            "OSL": [7, 10],
            "MYO": [1, 2, 3, 4, 5, 6],
            "UFL": [1, 2, 4, 5],
            "EUT": [3, 6],
            "INS": list(range(1, 27)),
            "JHU": list(range(24, 35)),
            "TSH": list(range(2, 502)),
        }
    elif data_type == "phantom":
        filelist = {
            "OSL": [2, 3, 4, 5, 6, 7],
            "MYO": [1, 2, 3, 4, 5, 6],
            "UFL": [1, 2, 3, 4, 5],
            "EUT": [1, 2, 3, 4, 5, 6],
            "INS": list(range(1, 27)),
        }
    elif data_type == "postcubdl":
        filelist = {"JHU": list(range(24, 35))}
    elif data_type == "invivo":
        print("Warning: TSH has 500 files. This will take a while.")
        filelist = {"JHU": list(range(24, 35)), "TSH": list(range(2, 502))}
    elif data_type == "simulation":
        filelist = {"OSL": [10]}
    elif data_type == "task1":
        filelist = {
            "PICMUS": [1, 2, 3, 4],
            "OSL": [7, 10],
            "TSH": random.sample(range(2, 500), 56),
            "MYO": [1, 2, 3, 4, 5],
            "UFL": [1, 2, 4, 5],
            "EUT": [3, 6],
            "INS": [4, 6, 8, 15, 16, 19, 21],
        }
    elif data_type == "task2": #6 + 50 + 5 + 11 + 2 + 5 = 79
        filelist = {
            "PICMUS": [0, 1, 2, 3, 4, 5],
            #"OSL": [7, 10],
            "TSH": random.sample(range(2, 500), 50),
            "MYO": random.sample(range(1, 7), 5),
            "JHU": list(range(24, 35)),
            "UFL": [1, 2], #[1, 2, 4, 5]
            "EUT": [3, 6],
            "INS": [6, 7, 8, 20, 21],#random.sample(range(1, 27), 5),
        }
    elif data_type == "task3":
        filelist = {
            "PICMUS": [0, 1, 2, 3, 4, 5],
            #"OSL": [7, 10],
            "TSH": list(range(2, 52)),
            "MYO": list(range(1, 6)),
            "JHU": list(range(24, 35)),
            "UFL": [1, 2],  # [1, 2, 4, 5]
            "EUT": [3, 6],
            "INS": [6, 7, 8, 20, 21],  # random.sample(range(1, 27), 5),
        }
    else:
        filelist = {
            "PICMUS": [1, 2, 3, 4],
            "OSL": [7, 10],
            "TSH": [2],
            "MYO": [1, 2, 3, 4, 5],
            "UFL": [1, 2, 4, 5],
            "INS": [4, 6, 8, 15, 16, 19, 21],
        }

    return filelist
