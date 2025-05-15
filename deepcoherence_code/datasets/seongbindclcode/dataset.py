# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset
# class IQDataset(Dataset):
#     def __init__(
#         self,
#         candidate_file_paths,
#         train=False,
#         validation=False,
#         inference=False,
#         horizontal_flip_prob=0.5,
#     ):
#         self.candidate_file_paths = candidate_file_paths
#         self.train = train
#         self.validation = validation
#         self.inference = inference
#         self.horizontal_flip_prob = horizontal_flip_prob
#         self.crop_size = (256, 256)

#         # 크롭 위치 저장용
#         self.crop_locs = {}  # idx → (top, left)

#     def __len__(self):
#         return len(self.candidate_file_paths)

#     def _random_horizontal_flip(self, data):
#         if torch.rand(1).item() < self.horizontal_flip_prob:
#             data = torch.flip(data, dims=[3])
#         return data

#     def __getitem__(self, idx):
#         file_path = self.candidate_file_paths[idx]
#         candidate = torch.load(file_path, map_location=torch.device("cpu"))
#         padded, orig_shape = self._apply_center_pad(candidate)
#         k, c, H_pad, W_pad = padded.shape

#         if self.inference:
#             return padded

#         elif self.train:
#             mid_idx = k // 2
#             data_wo_center = (
#                 torch.cat([padded[:mid_idx], padded[mid_idx + 1 :]], dim=0)
#                 if k > 1
#                 else padded
#             )
#             # 랜덤 크롭 위치 생성 및 저장
#             top, left = self._get_random_crop_locs(orig_shape)
#             self.crop_locs[idx] = (top, left)
#             cropped = self._crop_from_location(data_wo_center, top, left)
#             cropped = self._random_horizontal_flip(cropped)
#             final_data = self._pad_or_trim_to_N(cropped, N=74)
#             return final_data

#         elif self.validation:
#             # 같은 위치에서 크롭
#             top, left = self.crop_locs.get(idx, self._get_random_crop_locs(orig_shape))
#             cropped = self._crop_from_location(padded, top, left)
#             cropped = self._random_horizontal_flip(cropped)

#             kprime = cropped.size(0)
#             mid_idx = kprime // 2
#             center_angle = cropped[mid_idx].unsqueeze(0)
#             others = (
#                 torch.cat([cropped[:mid_idx], cropped[mid_idx + 1 :]], dim=0)
#                 if kprime > 1
#                 else torch.zeros((0, 2, 256, 256), dtype=cropped.dtype)
#             )
#             others_padded = self._pad_or_trim_to_N(others, N=73)
#             final_data = torch.cat([center_angle, others_padded], dim=0)
#             return final_data

#         return padded

#     def _apply_center_pad(self, data):
#         k, c, H, W = data.shape
#         orig_shape = (H, W)
#         processed = [
#             self._center_pad_or_crop_single(data[i]).unsqueeze(0) for i in range(k)
#         ]
#         padded = torch.cat(processed, dim=0)
#         return padded, orig_shape

#     def _center_pad_or_crop_single(self, img):
#         target = 1024
#         c, H, W = img.shape
#         # H 방향
#         if H > target:
#             top = (H - target) // 2
#             img = img[:, top : top + target, :]
#         elif H < target:
#             pad_top = (target - H) // 2
#             pad_bottom = target - H - pad_top
#             img = F.pad(img, (0, 0, pad_top, pad_bottom), mode="constant", value=0)
#         # W 방향
#         if W > target:
#             left = (W - target) // 2
#             img = img[:, :, left : left + target]
#         elif W < target:
#             pad_left = (target - W) // 2
#             pad_right = target - W - pad_left
#             img = F.pad(img, (pad_left, pad_right, 0, 0), mode="constant", value=0)
#         return img

#     def _get_random_crop_locs(self, orig_shape):
#         crop_h, crop_w = self.crop_size
#         H_orig, W_orig = orig_shape
#         pad_top = (1024 - H_orig) // 2
#         pad_left = (1024 - W_orig) // 2
#         top = pad_top + torch.randint(0, max(1, H_orig - crop_h + 1), (1,)).item()
#         left = pad_left + torch.randint(0, max(1, W_orig - crop_w + 1), (1,)).item()
#         return top, left

#     def _crop_from_location(self, data, top, left):
#         crop_h, crop_w = self.crop_size
#         cropped = data[:, :, top : top + crop_h, left : left + crop_w]
#         _, _, ch, cw = cropped.shape
#         if ch < crop_h or cw < crop_w:
#             cropped = F.pad(
#                 cropped, (0, crop_w - cw, 0, crop_h - ch), mode="constant", value=0
#             )
#         return cropped

#     def _pad_or_trim_to_N(self, data, N):
#         k, c, hh, ww = data.shape
#         if k < N:
#             pad_zeros = torch.zeros((N - k, c, hh, ww), dtype=data.dtype)
#             out = torch.cat([data, pad_zeros], dim=0)
#         elif k > N:
#             out = data[torch.randperm(k)[:N]]
#         else:
#             out = data
#         return out


import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class IQDataset(Dataset):
    """
    train=True:
      - 각 data (k,2,1024,1024)에서 중앙 인덱스(0도) 제거 → (k-1,2,1024,1024)
      - 제로패딩 영역은 무시하고, 원본 H, W 영역 내에서 256×256 랜덤 크롭 (angle 수 74 맞춤)

    validation=True:
      - 각 data (k,2,1024,1024)에서 원본 영역 내에서 256×256 랜덤 크롭
      - 중앙 angle은 index=0, 나머지 1~73에 배치 → 최종 (74,2,256,256) 반환

    inference=True:
      - 각 data (k,2,1024,1024)는 중앙 기준 패딩된 상태로 그대로 반환 (크롭 없음)
    """

    def __init__(
        self,
        candidate_file_paths,
        train=False,
        validation=False,
        inference=False,
        horizontal_flip_prob=0.5,  # 일단, horizontal flip 비활성화
    ):
        # 파일 경로 목록만 저장하여 메모리 사용 최소화
        self.candidate_file_paths = candidate_file_paths
        self.train = train
        self.validation = validation
        self.inference = inference
        self.horizontal_flip_prob = horizontal_flip_prob

        self.crop_size = (256, 256)

    def __len__(self):
        return len(self.candidate_file_paths)

    def _random_horizontal_flip(self, data):
        """
        Randomly flip the input tensor horizontally based on probability
        """
        if torch.rand(1).item() < self.horizontal_flip_prob:
            data = torch.flip(data, dims=[3])
        return data

    def __getitem__(self, idx):
        # 매번 디스크에서 한 샘플만 로드
        file_path = self.candidate_file_paths[idx]
        candidate = torch.load(file_path, map_location=torch.device("cpu"))
        padded, orig_shape = self._apply_center_pad(candidate)

        k, c, H_pad, W_pad = padded.shape

        if self.inference:
            return padded
        elif self.train:
            mid_idx = k // 2
            if k > 1:
                data_wo_center = torch.cat(
                    [padded[:mid_idx], padded[mid_idx + 1 :]], dim=0
                )
            else:
                data_wo_center = padded

            cropped = self._random_crop_original(data_wo_center, orig_shape)
            cropped = self._random_horizontal_flip(cropped)
            final_data = self._pad_or_trim_to_N(cropped, N=74)
            return final_data
        elif self.validation:
            cropped = self._random_crop_original(padded, orig_shape)
            cropped = self._random_horizontal_flip(cropped)

            kprime = cropped.size(0)
            mid_idx = kprime // 2
            center_angle = cropped[mid_idx].unsqueeze(0)
            if kprime > 1:
                others = torch.cat([cropped[:mid_idx], cropped[mid_idx + 1 :]], dim=0)
            else:
                others = torch.zeros(
                    (0, 2, 256, 256), dtype=cropped.dtype, device=cropped.device
                )
            others_padded = self._pad_or_trim_to_N(others, N=73)
            final_data = torch.cat([center_angle, others_padded], dim=0)
            return final_data

        return padded

    def _apply_center_pad(self, data):
        k, c, H, W = data.shape
        orig_shape = (H, W)
        processed = []
        for i in range(k):
            img = data[i]
            img_processed = self._center_pad_or_crop_single(img)
            processed.append(img_processed.unsqueeze(0))
        padded = torch.cat(processed, dim=0)
        return padded, orig_shape

    def _center_pad_or_crop_single(self, img):
        target = 1024
        c, H, W = img.shape
        # 세로 방향 패드/크롭
        if H > target:
            top = (H - target) // 2
            img = img[:, top : top + target, :]
        elif H < target:
            pad_total = target - H
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
            img = F.pad(img, (0, 0, pad_top, pad_bottom), mode="constant", value=0)
        # 가로 방향 패드/크롭
        if W > target:
            left = (W - target) // 2
            img = img[:, :, left : left + target]
        elif W < target:
            pad_total = target - W
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            img = F.pad(img, (pad_left, pad_right, 0, 0), mode="constant", value=0)
        return img

    def _random_crop_original(self, data, orig_shape):
        crop_h, crop_w = self.crop_size
        H_orig, W_orig = orig_shape
        pad_top = (1024 - H_orig) // 2
        pad_left = (1024 - W_orig) // 2
        if H_orig < crop_h or W_orig < crop_w:
            top = pad_top + (H_orig - crop_h) // 2 if H_orig >= crop_h else pad_top
            left = pad_left + (W_orig - crop_w) // 2 if W_orig >= crop_w else pad_left
        else:
            top = pad_top + torch.randint(0, H_orig - crop_h + 1, (1,)).item()
            left = pad_left + torch.randint(0, W_orig - crop_w + 1, (1,)).item()
        cropped = data[:, :, top : top + crop_h, left : left + crop_w]
        _, _, ch, cw = cropped.shape
        if ch < crop_h or cw < crop_w:
            pad_h = crop_h - ch
            pad_w = crop_w - cw
            cropped = F.pad(cropped, (0, pad_w, 0, pad_h), mode="constant", value=0)
        return cropped

    def _pad_or_trim_to_N(self, data, N):
        k, c, hh, ww = data.shape
        if k < N:
            pad_count = N - k
            pad_zeros = torch.zeros(
                (pad_count, c, hh, ww), dtype=data.dtype, device=data.device
            )
            out = torch.cat([data, pad_zeros], dim=0)
        elif k > N:
            indices = torch.randperm(k)[:N]
            out = data[indices]
        else:
            out = data
        return out


import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class FocusedIQDataset(Dataset):
    """
    Focused Beamformed 데이터용 Dataset 클래스

    - 각 샘플은 (N, 2, H, W) 형태의 tensor (.pt 파일)
    - scanline 수(N)는 그대로 유지
    - train/validation 모드에서는 각 scanline을 (2, 256, 256)으로 랜덤 크롭
    """

    def __init__(
        self,
        candidate_file_paths,
        train=False,
        validation=False,
        inference=False,
        horizontal_flip_prob=0.0,
    ):
        self.candidate_file_paths = candidate_file_paths
        self.train = train
        self.validation = validation
        self.inference = inference
        self.horizontal_flip_prob = horizontal_flip_prob
        self.crop_size = (256, 256)

    def __len__(self):
        return len(self.candidate_file_paths)

    def _random_horizontal_flip(self, data):
        if torch.rand(1).item() < self.horizontal_flip_prob:
            data = torch.flip(data, dims=[3])
        return data

    def __getitem__(self, idx):
        file_path = self.candidate_file_paths[idx]
        data = torch.load(file_path, map_location="cpu")  # (N, 2, H, W)
        N, C, H, W = data.shape

        if self.inference:
            padded = self._center_pad_batch(data)
            return padded  # (N, 2, 1024, 1024)

        # train or validation → 각 scanline마다 랜덤 크롭
        cropped_scanlines = []
        for i in range(N):
            scan = data[i]  # (2, H, W)
            scan = self._center_pad_or_crop_single(scan)  # (2, 1024, 1024)
            crop = self._random_crop_single(scan)  # (2, 256, 256)
            cropped_scanlines.append(crop.unsqueeze(0))  # (1, 2, 256, 256)

        cropped_data = torch.cat(cropped_scanlines, dim=0)  # (N, 2, 256, 256)
        cropped_data = self._random_horizontal_flip(cropped_data)
        return cropped_data

    def _center_pad_or_crop_single(self, img):
        target = 1024
        c, H, W = img.shape

        if H > target:
            top = (H - target) // 2
            img = img[:, top : top + target, :]
        elif H < target:
            pad_top = (target - H) // 2
            pad_bottom = target - H - pad_top
            img = F.pad(img, (0, 0, pad_top, pad_bottom), mode="constant", value=0)

        if W > target:
            left = (W - target) // 2
            img = img[:, :, left : left + target]
        elif W < target:
            pad_left = (target - W) // 2
            pad_right = target - W - pad_left
            img = F.pad(img, (pad_left, pad_right, 0, 0), mode="constant", value=0)

        return img

    def _random_crop_single(self, img):
        crop_h, crop_w = self.crop_size
        _, H, W = img.shape

        top = torch.randint(0, H - crop_h + 1, (1,)).item()
        left = torch.randint(0, W - crop_w + 1, (1,)).item()
        return img[:, top : top + crop_h, left : left + crop_w]

    def _center_pad_batch(self, data):
        padded = []
        for i in range(data.size(0)):
            padded_img = self._center_pad_or_crop_single(data[i])
            padded.append(padded_img.unsqueeze(0))
        return torch.cat(padded, dim=0)  # (N, 2, 1024, 1024)


# #filename : dataset.py
# #name : sean
# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset
# import numpy as np


# class IQDataset(Dataset):
#     """
#     train=True:
#       - 각 data (k,2,1024,1024)에서 중앙 인덱스(0도) 제거 → (k-1,2,1024,1024)
#       - 제로패딩 영역은 무시하고, 원본 H, W 영역 내에서 256×256 랜덤 크롭 (angle 수 74 맞춤)

#     validation=True:
#       - 각 data (k,2,1024,1024)에서 원본 영역 내에서 256×256 랜덤 크롭
#       - 중앙 angle은 index=0, 나머지 1~73에 배치 → 최종 (74,2,256,256) 반환

#     inference=True:
#       - 각 data (k,2,1024,1024)는 중앙 기준 패딩된 상태로 그대로 반환 (크롭 없음)
#     """

#     def __init__(
#         self, candidate_file_paths, train=False, validation=False, inference=False
#     ):
#         # candidate_file_paths: list of .pt file paths
#         self.candidate_file_paths = candidate_file_paths
#         self.train = train
#         self.validation = validation
#         self.inference = inference

#         self.crop_size = (256, 256)
#         # 캐시: index -> (padded_tensor, orig_shape)
#         self.center_pad_cache = {}

#     def __len__(self):
#         return len(self.candidate_file_paths)

#     def __getitem__(self, idx):
#         if idx not in self.center_pad_cache:
#             file_path = self.candidate_file_paths[idx]
#             candidate = torch.load(file_path, map_location=torch.device("cpu"))
#             padded, orig_shape = self._apply_center_pad(candidate)
#             self.center_pad_cache[idx] = (padded, orig_shape)
#         else:
#             padded, orig_shape = self.center_pad_cache[idx]

#         k, c, H_pad, W_pad = padded.shape

#         if self.inference:
#             return padded
#         elif self.train:
#             mid_idx = k // 2
#             if k > 1:
#                 data_wo_center = torch.cat(
#                     [padded[:mid_idx], padded[mid_idx + 1 :]], dim=0
#                 )
#             else:
#                 data_wo_center = padded
#             cropped = self._random_crop_original(data_wo_center, orig_shape)
#             final_data = self._pad_or_trim_to_N(cropped, N=74)
#             return final_data
#         elif self.validation:
#             cropped = self._random_crop_original(padded, orig_shape)
#             kprime = cropped.size(0)
#             mid_idx = kprime // 2
#             center_angle = cropped[mid_idx].unsqueeze(0)
#             if kprime > 1:
#                 others = torch.cat([cropped[:mid_idx], cropped[mid_idx + 1 :]], dim=0)
#             else:
#                 others = torch.zeros(
#                     (0, 2, 256, 256), dtype=cropped.dtype, device=cropped.device
#                 )
#             others_padded = self._pad_or_trim_to_N(others, N=73)
#             final_data = torch.cat([center_angle, others_padded], dim=0)
#             return final_data
#         return padded

#     def _apply_center_pad(self, data):
#         # data: (k,2,H,W)
#         k, c, H, W = data.shape
#         orig_shape = (H, W)
#         processed = []
#         for i in range(k):
#             img = data[i]
#             img_processed = self._center_pad_or_crop_single(img)
#             processed.append(img_processed.unsqueeze(0))
#         padded = torch.cat(processed, dim=0)
#         return padded, orig_shape

#     def _center_pad_or_crop_single(self, img):
#         target = 1024
#         c, H, W = img.shape
#         if H > target:
#             top = (H - target) // 2
#             img = img[:, top : top + target, :]
#         elif H < target:
#             pad_total = target - H
#             pad_top = pad_total // 2
#             pad_bottom = pad_total - pad_top
#             img = F.pad(img, (0, 0, pad_top, pad_bottom), mode="constant", value=0)
#         if W > target:
#             left = (W - target) // 2
#             img = img[:, :, left : left + target]
#         elif W < target:
#             pad_total = target - W
#             pad_left = pad_total // 2
#             pad_right = pad_total - pad_left
#             img = F.pad(img, (pad_left, pad_right, 0, 0), mode="constant", value=0)
#         return img

#     def _random_crop_original(self, data, orig_shape):
#         # data: (k,2,1024,1024), orig_shape: (H_orig, W_orig)
#         crop_h, crop_w = self.crop_size
#         H_orig, W_orig = orig_shape
#         pad_top = (1024 - H_orig) // 2
#         pad_left = (1024 - W_orig) // 2
#         if H_orig < crop_h or W_orig < crop_w:
#             top = pad_top + (H_orig - crop_h) // 2 if H_orig >= crop_h else pad_top
#             left = pad_left + (W_orig - crop_w) // 2 if W_orig >= crop_w else pad_left
#         else:
#             top = pad_top + torch.randint(0, H_orig - crop_h + 1, (1,)).item()
#             left = pad_left + torch.randint(0, W_orig - crop_w + 1, (1,)).item()
#         cropped = data[:, :, top : top + crop_h, left : left + crop_w]
#         _, _, ch, cw = cropped.shape
#         if ch < crop_h or cw < crop_w:
#             pad_h = crop_h - ch
#             pad_w = crop_w - cw
#             cropped = F.pad(cropped, (0, pad_w, 0, pad_h), mode="constant", value=0)
#         return cropped

#     def _pad_or_trim_to_N(self, data, N):
#         k, c, hh, ww = data.shape
#         if k < N:
#             pad_count = N - k
#             pad_zeros = torch.zeros(
#                 (pad_count, c, hh, ww), dtype=data.dtype, device=data.device
#             )
#             out = torch.cat([data, pad_zeros], dim=0)
#         elif k > N:
#             indices = torch.randperm(k)[:N]
#             out = data[indices]
#         else:
#             out = data
#         return out


# #--filename:dataset.py
# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset


# # 학습 시 사용할 random crop 함수
# def random_crop(iq_data, crop_size):
#     _, _, height, width = iq_data.shape
#     crop_h, crop_w = crop_size

#     top = torch.randint(0, height - crop_h + 1, (1,)).item()
#     left = torch.randint(0, width - crop_w + 1, (1,)).item()

#     cropped_data = iq_data[:, :, top : top + crop_h, left : left + crop_w]
#     return cropped_data


# # IQDataset: 학습 시에는 random crop, inference/validation 시에는 1024×1024 패딩 적용
# class IQDataset(Dataset):
#     def __init__(self, data_list, crop_size, validation=False):
#         """
#         :param data_list: List of data to load (각 data의 shape은 (k, 2, H, W))
#         :param crop_size: 학습 시 random crop에 사용할 (crop_h, crop_w)
#         :param validation: True이면 inference/validation용, False이면 학습용
#         """
#         self.data_list = data_list
#         self.crop_size = crop_size
#         self.validation = validation

#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, idx):
#         data = self.data_list[idx]  # (k, 2, H, W)
#         middle_idx = data.size(0) // 2
#         center_x = data[middle_idx]  # (2, H, W)
#         center_wo_data = torch.cat(
#             (data[:middle_idx], data[middle_idx + 1 :]), dim=0
#         )  # (k-1, 2, H, W)

#         if not self.validation:
#             # 학습 모드: center_wo_data에 대해 random crop 적용
#             _, _, h, w = center_wo_data.size()
#             crop_h, crop_w = self.crop_size

#             # 이미지 크기가 crop size보다 작으면 오른쪽과 아래쪽에 제로 패딩
#             pad_h = max(crop_h - h, 0)
#             pad_w = max(crop_w - w, 0)
#             if pad_h > 0 or pad_w > 0:
#                 center_wo_data = F.pad(center_wo_data, (0, pad_w, 0, pad_h))
#                 h, w = center_wo_data.size(2), center_wo_data.size(3)

#             top = torch.randint(0, h - crop_h + 1, (1,)).item()
#             left = torch.randint(0, w - crop_w + 1, (1,)).item()
#             cropped_data = center_wo_data[
#                 :, :, top : top + crop_h, left : left + crop_w
#             ]

#             # 각 sample이 항상 74 각도를 갖도록 조정
#             num_angles = cropped_data.size(0)
#             if num_angles < 74:
#                 pad_count = 74 - num_angles
#                 padding = torch.zeros(
#                     pad_count,
#                     cropped_data.size(1),
#                     cropped_data.size(2),
#                     cropped_data.size(3),
#                     device=cropped_data.device,
#                 )
#                 cropped_data = torch.cat((cropped_data, padding), dim=0)
#             elif num_angles > 74:
#                 indices = torch.randperm(num_angles)[:74]
#                 cropped_data = cropped_data[indices]

#             return cropped_data
#         else:
#             # inference/validation 모드: center_x와 center_wo_data에 대해 1024×1024 패딩 적용
#             original_h, original_w = center_wo_data.size(2), center_wo_data.size(3)
#             pad_height = max(1024 - original_h, 0)
#             pad_width = max(1024 - original_w, 0)

#             center_wo_y = F.pad(center_wo_data, (0, pad_width, 0, pad_height))
#             center_x = F.pad(center_x, (0, pad_width, 0, pad_height))

#             # validation 모드에서도 각도 수를 74로 고정
#             num_angles = center_wo_y.size(0)
#             if num_angles < 74:
#                 pad_count = 74 - num_angles
#                 padding = torch.zeros(
#                     pad_count,
#                     center_wo_y.size(1),
#                     center_wo_y.size(2),
#                     center_wo_y.size(3),
#                     device=center_wo_y.device,
#                 )
#                 center_wo_y = torch.cat((center_wo_y, padding), dim=0)
#             elif num_angles > 74:
#                 indices = torch.randperm(num_angles)[:74]
#                 center_wo_y = center_wo_y[indices]

#             return center_x, center_wo_y, original_h, original_w


# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset


# #######################################################
# # 1) 학습용: IQDatasetTrainRemoveMid
# #######################################################
# class IQDatasetTrainRemoveMid(Dataset):
#     """
#     (k,2,H,W)에서 중앙 인덱스(mid_idx = k//2) 제거 => (k-1,2,H,W).
#     이후 원본 '학습' 로직:
#       - 256x256 크롭/패딩
#       - (k-1)이 74개 미만이면 추가 패딩
#     최종 shape: (74,2,256,256)
#     """

#     def __init__(self, data_list, crop_size=(256, 256)):
#         super().__init__()
#         self.data_list = data_list
#         self.crop_size = crop_size

#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, idx):
#         data = self.data_list[idx]  # (k,2,H,W)
#         k, _, H, W = data.shape
#         mid_idx = k // 2  # 0도 인덱스

#         # 1) 중앙 인덱스 제거 => (k-1,2,H,W)
#         if k <= 1:
#             train_angles = torch.zeros((0, 2, H, W), dtype=data.dtype)
#         else:
#             if mid_idx == 0:
#                 train_angles = data[1:]
#             elif mid_idx == k - 1:
#                 train_angles = data[: k - 1]
#             else:
#                 train_angles = torch.cat([data[:mid_idx], data[mid_idx + 1 :]], dim=0)
#             # shape: (k-1,2,H,W)

#         # 2) 원본 '학습' 로직 (256x256 패딩/크롭, 최대 74개)
#         target_h, target_w = self.crop_size
#         processed_angles = []
#         for angle in train_angles:
#             # angle: (2,H,W)
#             current_h, current_w = angle.shape[1], angle.shape[2]

#             # 패딩 계산
#             pad_h = max(target_h - current_h, 0)
#             pad_w = max(target_w - current_w, 0)
#             pad_top = pad_h // 2
#             pad_bottom = pad_h - pad_top
#             pad_left = pad_w // 2
#             pad_right = pad_w - pad_left

#             # 패딩 적용
#             padded = F.pad(
#                 angle,
#                 (pad_left, pad_right, pad_top, pad_bottom),
#                 mode="constant",
#                 value=0,
#             )

#             # 중앙 크롭 (이미지가 더 큰 경우)
#             _, h_pad, w_pad = padded.shape
#             start_h = (h_pad - target_h) // 2
#             start_w = (w_pad - target_w) // 2
#             cropped = padded[
#                 :, start_h : start_h + target_h, start_w : start_w + target_w
#             ]
#             processed_angles.append(cropped)

#         if len(processed_angles) > 0:
#             processed_data = torch.stack(processed_angles, dim=0)  # (n,2,256,256)
#         else:
#             processed_data = torch.zeros((0, 2, target_h, target_w), dtype=data.dtype)

#         # (k-1)이 74개 미만이면 74개로 맞춤
#         if processed_data.size(0) < 74:
#             pad_count = 74 - processed_data.size(0)
#             pad_zeros = torch.zeros(
#                 (pad_count, 2, target_h, target_w),
#                 dtype=processed_data.dtype,
#                 device=processed_data.device,
#             )
#             processed_data = torch.cat([processed_data, pad_zeros], dim=0)

#         # 최종: (74,2,256,256)
#         return processed_data


# #######################################################
# # 2) 검증용: IQDatasetValMidVsRest
# #######################################################
# class IQDatasetValMidVsRest(Dataset):
#     """
#     (k,2,H,W)에서 중앙 인덱스(mid_idx = k//2) => center_x,
#     나머지 => center_wo_data.
#     이후 원본 '검증' 로직:
#       - 1024 패딩
#     최종 shape:
#       center_x: (2, H+pad, W+pad)
#       center_wo_data: (k-1,2,H+pad,W+pad)
#       original_h, original_w
#     """

#     def __init__(self, data_list):
#         super().__init__()
#         self.data_list = data_list

#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, idx):
#         data = self.data_list[idx]  # (k,2,H,W)
#         k, _, H, W = data.shape
#         mid_idx = k // 2

#         if k <= 1:
#             # 각도가 1개 이하 => 타깃 없음
#             center_x = data[0] if k > 0 else torch.zeros((2, H, W), dtype=data.dtype)
#             center_wo_data = torch.zeros((0, 2, H, W), dtype=data.dtype)
#             original_h, original_w = H, W
#         else:
#             center_x = data[mid_idx]  # (2,H,W)
#             if mid_idx == 0:
#                 center_wo_data = data[1:]
#             elif mid_idx == k - 1:
#                 center_wo_data = data[: k - 1]
#             else:
#                 center_wo_data = torch.cat([data[:mid_idx], data[mid_idx + 1 :]], dim=0)

#             if center_wo_data.numel() > 0:
#                 original_h, original_w = (
#                     center_wo_data.shape[2],
#                     center_wo_data.shape[3],
#                 )
#             else:
#                 original_h, original_w = H, W

#         pad_h = 1024 - original_h
#         pad_w = 1024 - original_w

#         device_for_data = data.device if data.is_cuda else "cpu"

#         if center_wo_data.numel() > 0:
#             center_wo_data = center_wo_data.to(device_for_data)
#             center_wo_data = F.pad(center_wo_data, (0, pad_w, 0, pad_h))

#         center_x = center_x.to(device_for_data)
#         center_x = F.pad(center_x, (0, pad_w, 0, pad_h))

#         return center_x, center_wo_data, original_h, original_w
