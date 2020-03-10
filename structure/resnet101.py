import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import os
import utils.config as cfg


class resnet101_trained(nn.Module):

    def __init__(self):
        super(resnet101_trained, self).__init__()

        os.environ["TORCH_HOME"] = "./"
        self.pretrain_res_101 = torchvision.models.resnet101(pretrained=cfg.pre_trained)
        self.layer_list = list(self.pretrain_res_101.children())

        self.c3_module = self.layer_list[:6]  # til 6 children.
        self.c4_module = self.layer_list[6]
        self.c5_module = self.layer_list[7]

    def forward(self, x):
        step = x
        ## Conv3
        for layer in self.c3_module:
            step = layer(step)
        C3 = step

        # Conv4_stack
        for layer in self.c4_module:
            step = layer(step)
        C4 = step

        # Conv5_stack
        for layer in self.c5_module:
            step = layer(step)
        C5 = step

        """
        Pl has resolution 2^l lower than the input
        construct a pyramid with levels P3 through P7

        means need 1/8, 1/16, 1/32, 1/64, 1/128

        C1 : torch.Size([1, 64, H/2, W/2])
        C2 : torch.Size([1, 256, H/4, W/4])
        C3 : torch.Size([1, 512, H/8, W/8])
        C4 : torch.Size([1, 1024, H/16, W/16])
        C5 : torch.Size([1, 2048, H/32, W/32])

        {C2;C3;C4;C5} for conv2, conv3, conv4, and conv5 outputs, and note that they have strides of {4, 8, 16, 32} pixels with respect to the input image.
        """
        # NOTICE: need only c3, c4 ,c5
        return C3, C4, C5
