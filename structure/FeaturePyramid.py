import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from structure.resnet101 import resnet101_trained as res_trained
import os


class FeaturePyramidStructure(nn.Module):

    def __init__(self, device):
        super(FeaturePyramidStructure, self).__init__()
        self.res_101 = res_trained()


        self.res_101.to(device=device)
        self.fixed_dim = 256

        self.fix_conv1 = nn.Conv2d(2048, self.fixed_dim, kernel_size=1, stride=1)
        self.batch_norm1 = nn.BatchNorm2d(self.fixed_dim)
        self.fix_conv2 = nn.Conv2d(256, self.fixed_dim, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(self.fixed_dim)
        self.fix_conv3 = nn.Conv2d(1024, self.fixed_dim, kernel_size=1, stride=1)
        self.batch_norm3 = nn.BatchNorm2d(self.fixed_dim)
        self.fix_conv4 = nn.Conv2d(256, self.fixed_dim, kernel_size=3, stride=1, padding=1)
        self.batch_norm4 = nn.BatchNorm2d(self.fixed_dim)
        self.fix_conv5 = nn.Conv2d(512, self.fixed_dim, kernel_size=1, stride=1)
        self.batch_norm5 = nn.BatchNorm2d(self.fixed_dim)
        self.fix_conv6 = nn.Conv2d(256, self.fixed_dim, kernel_size=3, stride=1, padding=1)
        self.batch_norm6 = nn.BatchNorm2d(self.fixed_dim)
        self.fix_conv7 = nn.Conv2d(256, self.fixed_dim, kernel_size=3, stride=2, padding=1)
        self.batch_norm7 = nn.BatchNorm2d(self.fixed_dim)
        self.fix_conv8 = nn.Conv2d(256, self.fixed_dim, kernel_size=3, stride=2, padding=1)
        self.batch_norm8 = nn.BatchNorm2d(self.fixed_dim)

        self.upsample_2x = torch.nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x):
        c3, c4, c5 = self.res_101(x)

        """
        To start the iteration, we simply attach a 1x1 convolutional layer on C5 to produce the coarsest resolution map. 
        
        append 3x3 a convolution on each merged map to 
        generate the final feature map, which is to reduce the aliasing effect of upsampling.
        """

        # to same size of dim
        t_p5 = F.relu(self.batch_norm1(self.fix_conv1(c5)))  # 2048
        p5 = F.relu(self.batch_norm2(self.fix_conv2(t_p5)))

        t_p4 = torch.add(self.upsample_2x(t_p5), F.relu(self.batch_norm3(self.fix_conv3(c4))))  # t_256 256
        p4 = F.relu(self.batch_norm4(self.fix_conv4(t_p4)))  # 256

        t_p3 = torch.add(self.upsample_2x(t_p4), F.relu(self.batch_norm5(self.fix_conv5(c3))))  # 256 256
        p3 = F.relu(self.batch_norm6(self.fix_conv6(t_p3)))  # 256

        # P6 is simply a stride two subsampling of P5
        # Notice: in this code subsampling using conv
        p6 = F.relu(self.batch_norm7(self.fix_conv7(t_p5)))
        p7 = F.relu(self.batch_norm8(self.fix_conv8(p6)))

        # created {p3, p4, p5, p6, p7}

        return p3, p4, p5, p6, p7

# # test forward
# fps = FeaturePyramidStructure()
# a = torch.ones(1, 3, 512, 512)
# for i in range(len(fps(a))):
#     print(fps(a)[i].size())
