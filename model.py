import torch
from torch import nn
from torchvision import models
import numpy as np
from torch.nn import functional as F

from fusion_models.TIP_version import Fusion_TIP
from fusion_models.ACCV_verison import Fusion_ACCV
from fusion_models.CMX import Fusion_CMX

pretrained_net = models.resnet34(pretrained=True)
class FCN(nn.Module):
    def __init__(self, num_classes, hyper_parm):
        super(FCN, self).__init__()

        # RGB modality backbone: 4 stages
        self.RGB_stage0 = nn.Sequential(*list(pretrained_net.children())[:-6])  # stage0
        self.RGB_stage1 = nn.Sequential(*list(pretrained_net.layer1))  # stage1
        self.RGB_stage2 = nn.Sequential(*list(pretrained_net.layer2))  # stage2
        self.RGB_stage3 = nn.Sequential(*list(pretrained_net.layer3))  # stage3

        # Depth modality backbone: 4 stages
        self.D_stage0 = nn.Sequential(*list(pretrained_net.children())[:-6])  # stage0
        self.D_stage1 = nn.Sequential(*list(pretrained_net.layer1))  # stage1
        self.D_stage2 = nn.Sequential(*list(pretrained_net.layer2))  # stage2
        self.D_stage3 = nn.Sequential(*list(pretrained_net.layer3))  # stage3



        self.scores1 = nn.Conv2d(256, num_classes, 1)
        self.scores2 = nn.Conv2d(128, num_classes, 1)
        self.scores3 = nn.Conv2d(64, num_classes, 1)

        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 8, 4, 2, bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 8)  # use a bilinear kernel

        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_4x.weight.data = bilinear_kernel(num_classes, num_classes, 4)  # use a bilinear kernel

        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_2x.weight.data = bilinear_kernel(num_classes, num_classes, 4)  # use a bilinear kernel

        ################# plug the fusion model #################

        ####  Fusion model CSCA++ (TIP_version)  ####
        # hyper-parameter: patch_size

        self.hyper_parm =hyper_parm
        self.fusion = nn.ModuleList([
            Fusion_TIP(in_channels=64, patch_size=self.hyper_parm),
            Fusion_TIP(in_channels=128, patch_size=self.hyper_parm),
            Fusion_TIP(in_channels=256, patch_size=self.hyper_parm)
        ])

        # ####  Fusion model CSCA (ACCV_version)  ####
        # hyper-parameter: grop_size

        # self.fusion = nn.ModuleList([
        #     Fusion_ACCV(in_channels=64, group_size=4),
        #     Fusion_ACCV(in_channels=128, group_size=3),
        #     Fusion_ACCV(in_channels=256, group_size=2)
        # ])

        ####  Fusion model CMX  ####

        # self.fusion = nn.ModuleList([
        #     Fusion_CMX(in_channels=64, reduction=1, num_heads=1, norm_layer=nn.BatchNorm2d),
        #     Fusion_CMX(in_channels=128, reduction=1, num_heads=1, norm_layer=nn.BatchNorm2d),
        #     Fusion_CMX(in_channels=256, reduction=1, num_heads=1, norm_layer=nn.BatchNorm2d)
        # ])

    def forward(self, x):
        ### can plug the fusion model to any encoder stage ###
        # encoder,
        RGB0 = self.RGB_stage0(x[0])
        D0 = self.D_stage0(x[1])

        # plug fusion module after stage 1
        RGB1 = self.RGB_stage1(RGB0)
        D1 = self.D_stage1(D0)
        RGB1, D1, shared1 = self.fusion[0]([RGB1, D1])
        s1 = shared1 # 1/4

        # plug fusion module after stage 2
        RGB2 = self.RGB_stage2(RGB1)
        D2 = self.D_stage2(D1)
        RGB2, D2, shared2 = self.fusion[1]([RGB2, D2])
        s2 = shared2  # 1/8

        # plug fusion module after stage 3
        RGB3 = self.RGB_stage3(RGB2)
        D3 = self.D_stage3(D2)
        RGB3, D3, shared3 = self.fusion[2]([RGB3, D3])
        s3 = shared3  # 1/16

        # decoder
        s3 = self.scores1(s3)
        s3 = self.upsample_2x(s3)
        s2 = self.scores2(s2)
        s2 = s2 + s3

        s1 = self.scores3(s1)
        s2 = self.upsample_4x(s2)
        s = s1 + s2

        s = self.upsample_8x(s)
        return s


def bilinear_kernel(in_channels, out_channels, kernel_size):
    '''
    return a bilinear filter tensor
    '''
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)




