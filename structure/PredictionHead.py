import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.config as cfg


class PredictionHead(nn.Module):
    def __init__(self):
        super(PredictionHead, self).__init__()

        self._a = cfg.a  # number of anchors.
        self._num_classes = cfg.num_classes  # number of classes.

        self._feature_dims = 256  # input_feature dims.

        self._num_prototype = cfg.num_prototype  # number of prototypes.

        self.Conv1 = nn.Conv2d(in_channels=self._feature_dims, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.batch1 = nn.BatchNorm2d(256)

        self.Conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.batch2 = nn.BatchNorm2d(256)

        self.Conv_Class = nn.Conv2d(in_channels=256, out_channels=self._num_classes * self._a, kernel_size=3, stride=1,
                                    padding=1)

        self.Conv_Box = nn.Conv2d(in_channels=256, out_channels=4 * self._a, kernel_size=3, stride=1,
                                  padding=1)  # This '4' means anchor's coordinates information.

        # mask coefficient.
        self.Conv_Coef = nn.Conv2d(in_channels=256, out_channels=self._num_prototype * self._a, kernel_size=3, stride=1,
                                   padding=1)

    def forward(self, features):
        # Features : p3, p4, p5, p6, p7
        # Anchor pixel size = [24; 48; 96; 192; 384]
        # This anchor's centers are located correspond to the feature map size.
        # Aspect ratios : 1:2, 1:1, 2:1  => _a = 3

        head = self.Conv1(features)
        head = self.batch1(head)
        head = F.relu(head)
        head = self.Conv2(head)
        head = self.batch2(head)
        head = F.relu(head)

        # conv class
        con_class = self.Conv_Class(head)

        # conv box
        con_box = self.Conv_Box(head)

        # conv coefficient
        con_coef = self.Conv_Coef(head)

        # output feature's index means real area of anchors.

        return con_class, con_box, con_coef
