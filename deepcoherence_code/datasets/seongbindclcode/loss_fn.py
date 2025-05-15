import torch
import torch.nn as nn


class CoherenceLoss(nn.Module):
    def __init__(self):
        super(CoherenceLoss, self).__init__()

    def forward(self, x, y):
        if x.dim() == 4:
            x = x.unsqueeze(1)  # (B, 1, 2, H, W)
        B, K, C, H, W = y.shape

        losses = []
        for i in range(K):
            yi = y[:, i]  # (B, 2, H, W)
            is_padding = torch.all(yi == 0, dim=(1, 2, 3))  # (B,)

            xi = x[:, 0]
            dot = torch.sum(xi * yi, dim=(-2, -1))  # (B, 2)
            norm_x = torch.sqrt(torch.sum(xi * xi, dim=(-2, -1)) + 1e-8)
            norm_y = torch.sqrt(torch.sum(yi * yi, dim=(-2, -1)) + 1e-8)
            corr = torch.sum(dot, dim=1) / (torch.sum(norm_x * norm_y, dim=1))  # (B,)

            corr[is_padding] = 0.0
            losses.append(corr)

        mean_corr = torch.stack(losses, dim=0)  # (K, B)
        valid_mask = mean_corr != 0.0
        valid_counts = valid_mask.sum(dim=0).clamp(min=1)
        mean_corr = mean_corr.sum(dim=0) / valid_counts  # (B,)

        # ✅ 각 샘플별 손실 반환 (reduction='none'과 유사)
        return -mean_corr  # shape: (B,)


# # filename:loss_fn.py
# import torch
# import torch.nn as nn

# class CoherenceLoss(nn.Module):
#     def __init__(self):
#         super(CoherenceLoss, self).__init__()

#     def forward(self, x, y):
#         if x.dim() == 4:
#             x = x.unsqueeze(1)  # (B, 1, 2, H, W)
#         B, K, C, H, W = y.shape

#         losses = []
#         for i in range(K):
#             xi = x[:, 0]  # (B, 2, H, W)
#             yi = y[:, i]  # (B, 2, H, W)
#             dot = torch.sum(xi * yi, dim=(-2, -1))
#             norm_x = torch.sqrt(torch.sum(xi * xi, dim=(-2, -1)) + 1e-8)
#             norm_y = torch.sqrt(torch.sum(yi * yi, dim=(-2, -1)) + 1e-8)
#             corr = torch.sum(dot, dim=1) / (torch.sum(norm_x * norm_y, dim=1))
#             losses.append(corr)

#         mean_corr = torch.stack(losses, dim=0).mean(dim=0)
#         return -mean_corr.mean()


# class CoherenceLoss(nn.Module):
#     def __init__(self, threshold=1e-8):
#         super().__init__()
#         self.threshold = threshold

#     def forward(self, predictions, targets, return_valid_count=False):
#         """
#         predictions: (B,2,H,W) 또는 (B,1,2,H,W)
#         targets    : (B, k-1, 2, H, W) 혹은 (B,73,2,H,W)
#         - 복소 상관 계수를 음수화하여 최소화 => 상관 계수 최대화
#         """
#         if predictions.dim() == 4:
#             predictions = predictions.unsqueeze(1)

#         # predictions: (B,1,2,H,W)
#         a = predictions[:, 0, 0]  # (B,H,W)
#         b = predictions[:, 0, 1]  # (B,H,W)
#         B, k_minus_1, _, H, W = targets.size()

#         eps = 1e-8
#         losses = []
#         valid_counts = []

#         for b_idx in range(B):
#             sum_corr = torch.zeros(
#                 1, device=predictions.device, dtype=predictions.dtype
#             )
#             valid_angles = 0
#             for t in range(k_minus_1):
#                 # 패딩된 각도는 모두 0이면 건너뜀
#                 if torch.all(targets[b_idx, t] == 0):
#                     continue

#                 # 대상 텐서의 노름(norm)이 threshold보다 큰 경우에만 유효로 간주
#                 norm = torch.sqrt(torch.sum(targets[b_idx, t] ** 2))
#                 if norm < self.threshold:
#                     continue

#                 c = targets[b_idx, t, 0]  # (H,W)
#                 d = targets[b_idx, t, 1]  # (H,W)

#                 num = torch.sum(a[b_idx] * c + b[b_idx] * d)
#                 denom_pred = torch.sum(a[b_idx] ** 2 + b[b_idx] ** 2)
#                 denom_target = torch.sum(c**2 + d**2)
#                 denom = torch.sqrt(denom_pred * denom_target) + eps
#                 corr = num / denom
#                 sum_corr += corr
#                 valid_angles += 1

#             losses.append(sum_corr)  # (sum over valid angles)
#             valid_counts.append(valid_angles)

#         losses = torch.stack(losses).squeeze()  # shape: (B,)
#         valid_counts_tensor = torch.tensor(
#             valid_counts, device=predictions.device, dtype=predictions.dtype
#         )

#         if return_valid_count:
#             return losses, valid_counts_tensor
#         else:
#             # 원래 방식: 각 샘플별 평균을 구해서 음수화
#             sample_losses = []
#             for loss_val, count in zip(losses, valid_counts):
#                 if count > 0:
#                     sample_losses.append(-loss_val / count)
#                 else:
#                     sample_losses.append(
#                         torch.tensor(
#                             0.0, device=predictions.device, dtype=predictions.dtype
#                         )
#                     )
#             sample_losses = torch.stack(sample_losses)
#             return sample_losses
