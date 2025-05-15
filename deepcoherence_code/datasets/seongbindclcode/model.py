#filname : model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def init_weights(net, init_type="normal", gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0.2, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)


def clip_gradients_by_layers(
    model, encoder_clip_value=1.0, decoder_clip_value=1.0, final_clip_value=0.5
):
    """
    각 레이어 그룹별로 다른 gradient clipping 값을 적용합니다.

    Args:
        model: DCLUNet 모델 인스턴스
        encoder_clip_value: 인코더 레이어(Conv1-Conv5)에 적용할 클리핑 값
        decoder_clip_value: 디코더 레이어(Up5-Up_conv2)에 적용할 클리핑 값
        final_clip_value: 최종 출력 레이어(Conv_1x1)에 적용할 클리핑 값
    """
    # 인코더 레이어 클리핑
    encoder_layers = [model.Conv1, model.Conv2, model.Conv3, model.Conv4, model.Conv5]
    for layer in encoder_layers:
        for param in layer.parameters():
            if param.grad is not None:
                torch.nn.utils.clip_grad_norm_(param, encoder_clip_value)

    # 디코더 레이어 클리핑
    decoder_layers = [
        model.Up5,
        model.Up_conv5,
        model.Up4,
        model.Up_conv4,
        model.Up3,
        model.Up_conv3,
        model.Up2,
        model.Up_conv2,
    ]
    for layer in decoder_layers:
        for param in layer.parameters():
            if param.grad is not None:
                torch.nn.utils.clip_grad_norm_(param, decoder_clip_value)

    # 최종 출력 레이어 클리핑
    for param in model.Conv_1x1.parameters():
        if param.grad is not None:
            torch.nn.utils.clip_grad_norm_(param, final_clip_value)


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class DCLUNet(nn.Module):
    def __init__(self, img_ch=2, output_ch=2):
        super(DCLUNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=32)
        self.Conv2 = conv_block(ch_in=32, ch_out=64)
        self.Conv3 = conv_block(ch_in=64, ch_out=128)
        self.Conv4 = conv_block(ch_in=128, ch_out=256)
        self.Conv5 = conv_block(ch_in=256, ch_out=512)

        self.Up5 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)

        self.Up3 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)

        self.Up2 = up_conv(ch_in=64, ch_out=32)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(x4)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return F.tanh(d1)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import init


# def init_weights(net, init_type="normal", gain=0.02, a=0.1, mode="fan_in"):
#     def init_func(m):
#         classname = m.__class__.__name__
#         if hasattr(m, "weight") and (
#             classname.find("Conv") != -1 or classname.find("Linear") != -1
#         ):
#             if init_type == "normal":
#                 init.normal_(m.weight.data, 0.0, gain)
#             elif init_type == "xavier":
#                 init.xavier_normal_(m.weight.data, gain=gain)
#             elif init_type == "kaiming":
#                 init.kaiming_normal_(m.weight.data, a=a, mode=mode)
#             elif init_type == "orthogonal":
#                 init.orthogonal_(m.weight.data, gain=gain)
#             else:
#                 raise NotImplementedError(
#                     "initialization method [%s] is not implemented" % init_type
#                 )
#             if hasattr(m, "bias") and m.bias is not None:
#                 init.constant_(m.bias.data, 0.0)
#         elif classname.find("BatchNorm2d") != -1:
#             init.normal_(m.weight.data, 1.0, gain)
#             init.constant_(m.bias.data, 0.0)

#     print(f"initialize network with {init_type} (gain={gain}, a={a}, mode={mode})")
#     net.apply(init_func)


# class conv_block(nn.Module):
#     def __init__(self, ch_in, ch_out):
#         super(conv_block, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(ch_out),  # BatchNorm 추가
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(ch_out),  # BatchNorm 추가
#             nn.LeakyReLU(0.2, inplace=True),
#         )

#     def forward(self, x):
#         x = self.conv(x)
#         return x


# class up_conv(nn.Module):
#     def __init__(self, ch_in, ch_out):
#         super(up_conv, self).__init__()
#         self.up = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(ch_out),  # BatchNorm 추가
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(ch_out),  # BatchNorm 추가
#             nn.LeakyReLU(0.2, inplace=True),
#         )

#     def forward(self, x):
#         x = self.up(x)
#         return x


# class DCLUNet(nn.Module):
#     def __init__(self, img_ch=2, output_ch=2):
#         super(DCLUNet, self).__init__()

#         self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.Conv1 = conv_block(ch_in=img_ch, ch_out=32)
#         self.Conv2 = conv_block(ch_in=32, ch_out=64)
#         self.Conv3 = conv_block(ch_in=64, ch_out=128)
#         self.Conv4 = conv_block(ch_in=128, ch_out=256)
#         self.Conv5 = conv_block(ch_in=256, ch_out=512)

#         self.Up5 = up_conv(ch_in=512, ch_out=256)
#         self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

#         self.Up4 = up_conv(ch_in=256, ch_out=128)
#         self.Up_conv4 = conv_block(ch_in=256, ch_out=128)

#         self.Up3 = up_conv(ch_in=128, ch_out=64)
#         self.Up_conv3 = conv_block(ch_in=128, ch_out=64)

#         self.Up2 = up_conv(ch_in=64, ch_out=32)
#         self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

#         self.Conv_1x1 = nn.Conv2d(32, output_ch, kernel_size=1, stride=1, padding=0)

#     def forward(self, x):
#         # encoding path
#         x1 = self.Conv1(x)

#         x2 = self.Maxpool(x1)
#         x2 = self.Conv2(x2)

#         x3 = self.Maxpool(x2)
#         x3 = self.Conv3(x3)

#         x4 = self.Maxpool(x3)
#         x4 = self.Conv4(x4)

#         x5 = self.Maxpool(x4)
#         x5 = self.Conv5(x5)

#         d5 = self.Up5(x5)
#         d5 = torch.cat((x4, d5), dim=1)
#         d5 = self.Up_conv5(d5)

#         d4 = self.Up4(d5)  # 수정: x4 -> d5
#         d4 = torch.cat((x3, d4), dim=1)
#         d4 = self.Up_conv4(d4)

#         d3 = self.Up3(d4)
#         d3 = torch.cat((x2, d3), dim=1)
#         d3 = self.Up_conv3(d3)

#         d2 = self.Up2(d3)
#         d2 = torch.cat((x1, d2), dim=1)
#         d2 = self.Up_conv2(d2)

#         d1 = self.Conv_1x1(d2)

#         return F.tanh(d1)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import init


# def init_weights(net, init_type="normal", gain=0.02):
#     def init_func(m):
#         classname = m.__class__.__name__
#         if hasattr(m, "weight") and (
#             classname.find("Conv") != -1 or classname.find("Linear") != -1
#         ):
#             if init_type == "normal":
#                 init.normal_(m.weight.data, 0.0, gain)
#             elif init_type == "xavier":
#                 init.xavier_normal_(m.weight.data, gain=gain)
#             elif init_type == "kaiming":
#                 init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
#             elif init_type == "orthogonal":
#                 init.orthogonal_(m.weight.data, gain=gain)
#             else:
#                 raise NotImplementedError(
#                     "initialization method [%s] is not implemented" % init_type
#                 )
#             if hasattr(m, "bias") and m.bias is not None:
#                 init.constant_(m.bias.data, 0.0)
#         elif classname.find("BatchNorm2d") != -1:
#             init.normal_(m.weight.data, 1.0, gain)
#             init.constant_(m.bias.data, 0.0)

#     print("initialize network with %s" % init_type)
#     net.apply(init_func)


# class conv_block(nn.Module):
#     def __init__(self, ch_in, ch_out):
#         super(conv_block, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.LeakyReLU(0.2, inplace=True),
#         )

#     def forward(self, x):
#         x = self.conv(x)
#         return x


# class up_conv(nn.Module):
#     def __init__(self, ch_in, ch_out):
#         super(up_conv, self).__init__()
#         self.up = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.LeakyReLU(0.2, inplace=True),
#         )

#     def forward(self, x):
#         x = self.up(x)
#         return x


# class DCLUNet(nn.Module):
#     def __init__(self, img_ch=2, output_ch=2):
#         super(DCLUNet, self).__init__()

#         self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.Conv1 = conv_block(ch_in=img_ch, ch_out=32)
#         self.Conv2 = conv_block(ch_in=32, ch_out=64)
#         self.Conv3 = conv_block(ch_in=64, ch_out=128)
#         self.Conv4 = conv_block(ch_in=128, ch_out=256)
#         self.Conv5 = conv_block(ch_in=256, ch_out=512)

#         self.Up5 = up_conv(ch_in=512, ch_out=256)
#         self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

#         self.Up4 = up_conv(ch_in=256, ch_out=128)
#         self.Up_conv4 = conv_block(ch_in=256, ch_out=128)

#         self.Up3 = up_conv(ch_in=128, ch_out=64)
#         self.Up_conv3 = conv_block(ch_in=128, ch_out=64)

#         self.Up2 = up_conv(ch_in=64, ch_out=32)
#         self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

#         self.Conv_1x1 = nn.Conv2d(32, output_ch, kernel_size=1, stride=1, padding=0)

#     def forward(self, x):
#         # encoding path
#         x1 = self.Conv1(x)

#         x2 = self.Maxpool(x1)
#         x2 = self.Conv2(x2)

#         x3 = self.Maxpool(x2)
#         x3 = self.Conv3(x3)

#         x4 = self.Maxpool(x3)
#         x4 = self.Conv4(x4)

#         x5 = self.Maxpool(x4)
#         x5 = self.Conv5(x5)

#         d5 = self.Up5(x5)
#         d5 = torch.cat((x4, d5), dim=1)
#         d5 = self.Up_conv5(d5)

#         d4 = self.Up4(x4)
#         d4 = torch.cat((x3, d4), dim=1)
#         d4 = self.Up_conv4(d4)

#         d3 = self.Up3(d4)
#         d3 = torch.cat((x2, d3), dim=1)
#         d3 = self.Up_conv3(d3)

#         d2 = self.Up2(d3)
#         d2 = torch.cat((x1, d2), dim=1)
#         d2 = self.Up_conv2(d2)

#         d1 = self.Conv_1x1(d2)

#         return F.tanh(d1)


# #filename: model.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# def crop_to_match(skip, upsampled):
#     """
#     skip: skip connection에서 가져온 텐서 (B, C, H, W)
#     upsampled: 업샘플링 후의 텐서 (B, C, H_target, W_target)
#     두 텐서의 H, W가 다를 경우, skip 텐서를 중앙에서 H_target, W_target 크기로 crop합니다.
#     """
#     _, _, H, W = skip.shape
#     _, _, H_target, W_target = upsampled.shape
#     diff_h = H - H_target
#     diff_w = W - W_target
#     crop_top = diff_h // 2
#     crop_left = diff_w // 2
#     return skip[:, :, crop_top : crop_top + H_target, crop_left : crop_left + W_target]


# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.LeakyReLU(),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.LeakyReLU(),
#         )

#     def forward(self, x):
#         return self.double_conv(x)


# class DCLUNet(nn.Module):
#     def __init__(self, in_channels=2, out_channels=2):
#         super(DCLUNet, self).__init__()

#         self.enc1 = DoubleConv(in_channels, 32)
#         self.enc2 = DoubleConv(32, 64)
#         self.enc3 = DoubleConv(64, 128)
#         self.enc4 = DoubleConv(128, 256)
#         self.enc5 = DoubleConv(256, 512)

#         self.pool = nn.MaxPool2d(2)

#         self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
#         self.dec4 = DoubleConv(512, 256)

#         self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
#         self.dec3 = DoubleConv(256, 128)

#         self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
#         self.dec2 = DoubleConv(128, 64)

#         self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
#         self.dec1 = DoubleConv(64, 32)

#         self.final = nn.Conv2d(32, out_channels, kernel_size=1)
#         self.tanh = nn.Tanh()

#     def forward(self, x):
#         x1 = self.enc1(x)
#         x2 = self.enc2(self.pool(x1))
#         x3 = self.enc3(self.pool(x2))
#         x4 = self.enc4(self.pool(x3))
#         x5 = self.enc5(self.pool(x4))

#         x = self.up4(x5)
#         x4_crop = crop_to_match(x4, x)
#         x = self.dec4(torch.cat([x, x4_crop], dim=1))

#         x = self.up3(x)
#         x3_crop = crop_to_match(x3, x)
#         x = self.dec3(torch.cat([x, x3_crop], dim=1))

#         x = self.up2(x)
#         x2_crop = crop_to_match(x2, x)
#         x = self.dec2(torch.cat([x, x2_crop], dim=1))

#         x = self.up1(x)
#         x1_crop = crop_to_match(x1, x)
#         x = self.dec1(torch.cat([x, x1_crop], dim=1))
#         x = self.final(x)

#         return self.tanh(x)


# import torch
# import torch.nn as nn


# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.leaky_relu1 = nn.LeakyReLU(0.01, inplace=True)

#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.leaky_relu2 = nn.LeakyReLU(0.01, inplace=True)

#     def forward(self, x):
#         # Conv1 -> BN -> LeakyReLU
#         x = self.leaky_relu1(self.bn1(self.conv1(x)))
#         # Conv2 -> BN -> LeakyReLU
#         x = self.leaky_relu2(self.bn2(self.conv2(x)))
#         return x


# class UpConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
#         # 업샘플 후, skip connection과 concat -> ConvBlock
#         self.conv_block = ConvBlock(out_channels * 2, out_channels)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         # skip connection을 위해 x2를 x1 크기에 맞춰 crop
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]
#         x2 = x2[
#             :,
#             :,
#             diffY // 2 : x2.size()[2] - (diffY - diffY // 2),
#             diffX // 2 : x2.size()[3] - (diffX - diffX // 2),
#         ]
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv_block(x)


# class DCLUNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Encoder
#         self.enc1 = ConvBlock(2, 32)
#         self.enc2 = ConvBlock(32, 64)
#         self.enc3 = ConvBlock(64, 128)
#         self.enc4 = ConvBlock(128, 256)
#         self.enc5 = ConvBlock(256, 512)

#         # Decoder
#         self.dec4 = UpConvBlock(512, 256)
#         self.dec3 = UpConvBlock(256, 128)
#         self.dec2 = UpConvBlock(128, 64)
#         self.dec1 = UpConvBlock(64, 32)

#         self.final_conv = nn.Conv2d(32, 2, kernel_size=1)
#         self.pool = nn.MaxPool2d(2)
#         self.out_act = nn.Tanh()

#     def forward(self, x):
#         # Encoding path
#         e1 = self.enc1(x)
#         e2 = self.enc2(self.pool(e1))
#         e3 = self.enc3(self.pool(e2))
#         e4 = self.enc4(self.pool(e3))
#         e5 = self.enc5(self.pool(e4))

#         # Decoding path
#         d4 = self.dec4(e5, e4)
#         d3 = self.dec3(d4, e3)
#         d2 = self.dec2(d3, e2)
#         d1 = self.dec1(d2, e1)

#         out = self.final_conv(d1)
#         return self.out_act(out)


# 2
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import init


# def init_weights(net, init_type="normal", gain=0.02):
#     def init_func(m):
#         classname = m.__class__.__name__
#         if hasattr(m, "weight") and (
#             classname.find("Conv") != -1 or classname.find("Linear") != -1
#         ):
#             if init_type == "normal":
#                 init.normal_(m.weight.data, 0.0, gain)
#             elif init_type == "xavier":
#                 init.xavier_normal_(m.weight.data, gain=gain)
#             elif init_type == "kaiming":
#                 init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
#             elif init_type == "orthogonal":
#                 init.orthogonal_(m.weight.data, gain=gain)
#             else:
#                 raise NotImplementedError(
#                     "initialization method [%s] is not implemented" % init_type
#                 )
#             if hasattr(m, "bias") and m.bias is not None:
#                 init.constant_(m.bias.data, 0.0)
#         elif classname.find("BatchNorm2d") != -1:
#             init.normal_(m.weight.data, 1.0, gain)
#             init.constant_(m.bias.data, 0.0)

#     print("initialize network with %s" % init_type)
#     net.apply(init_func)


# class conv_block(nn.Module):
#     def __init__(self, ch_in, ch_out):
#         super(conv_block, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.LeakyReLU(0.2, inplace=True),
#         )

#     def forward(self, x):
#         return self.conv(x)


# class up_conv(nn.Module):
#     def __init__(self, ch_in, ch_out):
#         super(up_conv, self).__init__()
#         self.up = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.LeakyReLU(0.2, inplace=True),
#         )

#     def forward(self, x):
#         return self.up(x)


# class DCLUNet(nn.Module):
#     def __init__(self, img_ch=2, output_ch=2):
#         super(DCLUNet, self).__init__()

#         self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.Conv1 = conv_block(ch_in=img_ch, ch_out=32)
#         self.Conv2 = conv_block(ch_in=32, ch_out=64)
#         self.Conv3 = conv_block(ch_in=64, ch_out=128)
#         self.Conv4 = conv_block(ch_in=128, ch_out=256)
#         self.Conv5 = conv_block(ch_in=256, ch_out=512)

#         self.Up5 = up_conv(ch_in=512, ch_out=256)
#         self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

#         self.Up4 = up_conv(ch_in=256, ch_out=128)
#         self.Up_conv4 = conv_block(ch_in=256, ch_out=128)

#         self.Up3 = up_conv(ch_in=128, ch_out=64)
#         self.Up_conv3 = conv_block(ch_in=128, ch_out=64)

#         self.Up2 = up_conv(ch_in=64, ch_out=32)
#         self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

#         self.Conv_1x1 = nn.Conv2d(32, output_ch, kernel_size=1, stride=1, padding=0)

#     def forward(self, x):
#         # encoding path
#         x1 = self.Conv1(x)

#         x2 = self.Maxpool(x1)
#         x2 = self.Conv2(x2)

#         x3 = self.Maxpool(x2)
#         x3 = self.Conv3(x3)

#         x4 = self.Maxpool(x3)
#         x4 = self.Conv4(x4)

#         x5 = self.Maxpool(x4)
#         x5 = self.Conv5(x5)

#         # decoding path
#         d5 = self.Up5(x5)
#         d5 = torch.cat((x4, d5), dim=1)
#         d5 = self.Up_conv5(d5)

#         # 수정된 부분: x4 대신 d5의 결과를 사용하여 업샘플링
#         d4 = self.Up4(d5)
#         d4 = torch.cat((x3, d4), dim=1)
#         d4 = self.Up_conv4(d4)

#         d3 = self.Up3(d4)
#         d3 = torch.cat((x2, d3), dim=1)
#         d3 = self.Up_conv3(d3)

#         d2 = self.Up2(d3)
#         d2 = torch.cat((x1, d2), dim=1)
#         d2 = self.Up_conv2(d2)

#         d1 = self.Conv_1x1(d2)

#         return F.tanh(d1)


# import torch
# import torch.nn as nn


# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ConvBlock, self).__init__()

#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.leaky_relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

#     def forward(self, x):
#         x = self.leaky_relu(self.conv1(x))
#         x = self.leaky_relu(self.conv2(x))

#         return x


# class UpConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(UpConvBlock, self).__init__()

#         self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
#         self.conv_block = ConvBlock(in_channels, out_channels)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         x = torch.cat([x2, x1], dim=1)

#         return self.conv_block(x)


# class DCLUNet(nn.Module):
#     def __init__(self):
#         super(DCLUNet, self).__init__()

#         # Encoder
#         self.enc1 = ConvBlock(2, 32)
#         self.enc2 = ConvBlock(32, 64)
#         self.enc3 = ConvBlock(64, 128)
#         self.enc4 = ConvBlock(128, 256)
#         self.enc5 = ConvBlock(256, 512)

#         # Decoder
#         self.dec4 = UpConvBlock(512, 256)
#         self.dec3 = UpConvBlock(256, 128)
#         self.dec2 = UpConvBlock(128, 64)
#         self.dec1 = UpConvBlock(64, 32)

#         self.final_conv = nn.Conv2d(32, 2, kernel_size=1)

#         self.pool = nn.MaxPool2d(2)
#         self.out_act = nn.Tanh()

#     def forward(self, x):
#         # Encoder
#         e1 = self.enc1(x)
#         e2 = self.enc2(self.pool(e1))
#         e3 = self.enc3(self.pool(e2))
#         e4 = self.enc4(self.pool(e3))
#         e5 = self.enc5(self.pool(e4))

#         # Decoder
#         d4 = self.dec4(e5, e4)
#         d3 = self.dec3(d4, e3)
#         d2 = self.dec2(d3, e2)
#         d1 = self.dec1(d2, e1)

#         out = self.final_conv(d1)
#         out = self.out_act(out)

#         return out
