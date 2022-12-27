import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

class Fusion_ACCV(nn.Module):
    def __init__(self, in_channels, group_size):
        super(Fusion_ACCV, self).__init__()
        self.L = group_size
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

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # pyramid pooling
        pyramid_RGB = self.RGB_pyramid_msc(x[0])
        pyramid_D = self.D_pyramid_msc(x[1])

        # SCA Block
        adapt_channels = self.L * self.out_channels
        adapt_channels = int(adapt_channels)

        batch_size = pyramid_RGB.size(0)

        rgb_query = self.RGB_query(pyramid_RGB).view(batch_size, adapt_channels, -1).permute(0, 2, 1)

        rgb_key = self.RGB_key(pyramid_RGB).view(batch_size, adapt_channels, -1)
        rgb_value = self.RGB_value(pyramid_RGB).view(batch_size, adapt_channels, -1).permute(0, 2, 1)

        batch_size = pyramid_D.size(0)
        D_query = self.D_query(pyramid_D).view(batch_size, adapt_channels, -1).permute(0, 2, 1)
        D_key = self.D_key(pyramid_D).view(batch_size, adapt_channels, -1)
        D_value = self.D_value(pyramid_D).view(batch_size, adapt_channels, -1).permute(0, 2, 1)

        RGB_sim_map = torch.matmul(D_query, rgb_key)
        RGB_sim_map = (adapt_channels ** -.5) * RGB_sim_map
        RGB_sim_map = F.softmax(RGB_sim_map, dim=-1)
        RGB_context = torch.matmul(RGB_sim_map, rgb_value)
        RGB_context = RGB_context.permute(0, 2, 1).contiguous()
        RGB_context = RGB_context.view(batch_size, self.out_channels, *pyramid_RGB.size()[2:])
        RGB_context = self.RGB_W(RGB_context)

        D_sim_map = torch.matmul(rgb_query, D_key)
        D_sim_map = (adapt_channels ** -.5) * D_sim_map
        D_sim_map = F.softmax(D_sim_map, dim=-1)
        D_context = torch.matmul(D_sim_map, D_value)
        D_context = D_context.permute(0, 2, 1).contiguous()
        D_context = D_context.view(batch_size, self.out_channels, *pyramid_D.size()[2:])
        D_context = self.D_W(D_context)

        # CFA block
        cat_fea = torch.cat([D_context, RGB_context], dim=1)

        attention_vector_RGB = self.gate_RGB(cat_fea)
        attention_vector_D = self.gate_D(cat_fea)

        attention_vector = torch.cat([attention_vector_RGB, attention_vector_D], dim=1)
        attention_vector = self.softmax(attention_vector)
        attention_vector_RGB, attention_vector_D = attention_vector[:, 0:1, :, :], attention_vector[:, 1:2, :, :]
        new_shared = x[0] * attention_vector_RGB + x[1] * attention_vector_D  # feature aggretation

        new_RGB = (x[0] + new_shared) / 2
        new_D = (x[1] + new_shared) / 2

        new_RGB = self.relu1(new_RGB)
        new_D = self.relu2(new_D)

        return new_RGB, new_D, new_shared

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