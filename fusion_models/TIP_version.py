import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np

class Fusion_TIP(nn.Module):
    def __init__(self, in_channels, patch_size):
        super(Fusion_TIP, self).__init__()
        self.n_h, self.n_w = patch_size, patch_size
        self.seen = 0

        self.RGB_pyramid_msc = MSC(in_channels)
        self.D_pyramid_msc = MSC(in_channels)

        self.channels = in_channels
        self.out_channels = in_channels //2

        self.RGB_key = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.out_channels,
                      kernel_size=1, stride=1, padding=0), nn.Dropout(0.5),
            nn.BatchNorm2d(self.out_channels), nn.ReLU(),
        )
        self.RGB_query = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.out_channels,
                      kernel_size=1, stride=1, padding=0), nn.Dropout(0.5),
            nn.BatchNorm2d(self.out_channels), nn.ReLU(),
        )
        self.RGB_value = nn.Conv2d(in_channels=self.channels, out_channels=self.out_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.RGB_W = nn.Conv2d(in_channels=self.out_channels, out_channels=self.channels,
                           kernel_size=1, stride=1, padding=0)

        self.D_key = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.out_channels,
                      kernel_size=1, stride=1, padding=0), nn.Dropout(0.5),
            nn.BatchNorm2d(self.out_channels), nn.ReLU(),
        )
        self.D_query = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.out_channels,
                      kernel_size=1, stride=1, padding=0), nn.Dropout(0.5),
            nn.BatchNorm2d(self.out_channels), nn.ReLU(),
        )
        self.D_value = nn.Conv2d(in_channels=self.channels, out_channels=self.out_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.D_W = nn.Conv2d(in_channels=self.out_channels, out_channels=self.channels,
                           kernel_size=1, stride=1, padding=0)

        self.gate_RGB = nn.Conv2d(self.channels * 2, 1, kernel_size=1, bias=True)
        self.gate_D = nn.Conv2d(self.channels * 2, 1, kernel_size=1, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.softmax = nn.Softmax(dim=1)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()


    def forward(self, x):
        # pyramid pooling
        pyramid_RGB = self.RGB_pyramid_msc(x[0])
        pyramid_D = self.D_pyramid_msc(x[1])

        # across attention with spacial partition
        # SCA++
        feature_size, channels = pyramid_RGB.size()[2:], pyramid_RGB.size(1) // 2
        rgb_query, _ = self.spacial_split(self.RGB_query(pyramid_RGB))
        rgb_query = rgb_query.permute(0, 2, 1)
        rgb_key, patch_param = self.spacial_split(self.RGB_key(pyramid_RGB))
        rgb_value, _ = self.spacial_split(self.RGB_value(pyramid_RGB))
        rgb_value = rgb_value.permute(0, 2, 1)
        D_query, _ = self.spacial_split(self.D_query(pyramid_D))
        D_query = D_query.permute(0, 2, 1)#[BP, N, C]
        D_key, _ = self.spacial_split(self.D_key(pyramid_D))#[BP, C, N]
        D_value, _ = self.spacial_split(self.D_value(pyramid_D))
        D_value = D_value.permute(0, 2, 1)#[BP, N, C]


        RGB_sim_map = torch.matmul(D_query, rgb_key)#[BP, N, N]
        RGB_sim_map = (channels ** -.5) * RGB_sim_map
        RGB_sim_map = F.softmax(RGB_sim_map, dim=-1)
        RGB_context = torch.matmul(RGB_sim_map, rgb_value)#[BP, N, C]
        RGB_context = self.spacial_splice(RGB_context, patch_param)
        RGB_context = self.RGB_W(RGB_context)

        D_sim_map = torch.matmul(rgb_query, D_key)
        D_sim_map = (channels ** -.5) * D_sim_map
        D_sim_map = F.softmax(D_sim_map, dim=-1)
        D_context = torch.matmul(D_sim_map, D_value)
        D_context = self.spacial_splice(D_context, patch_param)
        D_context = self.D_W(D_context)

        # CFA Block
        cat_fea = torch.cat([D_context, RGB_context], dim=1)

        attention_vector_RGB = self.gate_RGB(cat_fea)
        attention_vector_D = self.gate_D(cat_fea)

        attention_vector = torch.cat([attention_vector_RGB, attention_vector_D], dim=1)
        attention_vector = self.softmax(attention_vector)  # attention vector in paper
        attention_vector_RGB, attention_vector_D = attention_vector[:, 0:1, :, :], attention_vector[:, 1:2, :, :]
        new_shared = x[0] * attention_vector_RGB + x[1] * attention_vector_D  # feature aggretation

        new_RGB = (x[0] + new_shared) / 2
        new_D = (x[1] + new_shared) / 2

        new_RGB = self.relu1(new_RGB)
        new_D = self.relu2(new_D)

        return new_RGB, new_D, new_shared

    def spacial_split(self, Fea):
        batch, channels, H, W = Fea.shape
        num_patches = self.n_h*self.n_w
        new_H = int(math.ceil(H / self.n_h) * self.n_h)
        new_W = int(math.ceil(W / self.n_w) * self.n_w)

        interpolate = False
        if new_H != H or new_W != W:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            Fea = F.interpolate(Fea, size=(new_H, new_W), mode="bilinear", align_corners=False)
            interpolate = True
        patch_h = new_H //self.n_h
        patch_w = new_W //self.n_w
        patch_unit = patch_h * patch_w


        # [B, C, H, W] --> [B * C * n_h, p_h, n_w, p_w]
        reshaped_Fea = Fea.reshape(batch * channels * self.n_h, patch_h, self.n_w, patch_w)
        # [B * C * n_h, p_h, n_w, p_w] --> [B * C * n_h, n_w, p_h, p_w]
        transposed_Fea = reshaped_Fea.transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] --> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        reshaped_Fea = transposed_Fea.reshape(batch, channels, num_patches, patch_unit)
        # [B, C, N, P] --> [B, P, C, N]
        transposed_Fea = reshaped_Fea.permute(0, 3, 1, 2)
        # [B, P, C, N] --> [BP, C, N]
        patches = transposed_Fea.reshape(batch * patch_unit, channels, -1)

        return patches, [batch, channels, H, W, patch_h, patch_w]

    def spacial_splice(self, patches, patch_param):
        [batch_size, channels, H, W, patch_h, patch_w] = patch_param
        patch_unit = patch_h * patch_w
        num_patches = self.n_h*self.n_w

        # [BP, N, C] --> [B, P, N, C]
        patches = patches.reshape(batch_size, patch_unit, num_patches, -1)
        # [B, P, N, C] --> [B, C, N, P]
        patches = patches.permute(0, 3, 2, 1).contiguous()

        # [B, C, N, P] --> [B*C*n_h, n_w, p_h, p_w]
        Fea = patches.reshape(batch_size * channels * self.n_h, self.n_w, patch_h, patch_w)
        # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w]
        Fea = Fea.transpose(1, 2)
        # [B*C*n_h, p_h, n_w, p_w] --> [B, C, new_H, new_W]
        Fea = Fea.reshape(batch_size, channels, self.n_h * patch_h, self.n_w * patch_w)
        if self.n_h * patch_h != H or self.n_w * patch_w != W:
            Fea = F.interpolate(Fea, size=(H, W), mode="bilinear", align_corners=False)
        return Fea

class MSC(nn.Module):
    def __init__(self, channels):
        super(MSC, self).__init__()
        self.channels = channels
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.conv = nn.Sequential(
            nn.Conv2d(3*channels, channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = nn.functional.interpolate(self.pool1(x), x.shape[2:])
        x2 = nn.functional.interpolate(self.pool2(x), x.shape[2:])
        concat = torch.cat([x, x1, x2], 1)
        fusion = self.conv(concat)
        return fusion