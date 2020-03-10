import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.config as cfg


class ProtoNet(nn.Module):
    def __init__(self):
        super(ProtoNet, self).__init__()

        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.batch1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.batch2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.batch3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.batch4 = nn.BatchNorm2d(256)
        self.deconv1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, bias=True)
        self.de_batch1 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, cfg.num_prototype, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        layer = F.relu(self.batch1(self.conv1(x)))
        layer = F.relu(self.batch2(self.conv2(layer)))
        layer = F.relu(self.batch3(self.conv3(layer)))
        layer = F.relu(self.batch4(self.conv4(layer)))
        layer = F.relu(self.de_batch1(self.deconv1(layer)))
        proto_types = self.conv5(layer)

        """
        In last layer.
        'deconvs are 2x2 with stride 2, and we use ReLU in hidden layers.'
        -mask rcnn-
        """

        return proto_types

# # test forward
# pro = ProtoNet()
# a = torch.ones(1, 3, 69, 69)
# result = pro(a)
# print(result.size())
