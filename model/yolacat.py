import torch
import torch.nn as nn
import torch.nn.functional as F
from structure.FeaturePyramid import FeaturePyramidStructure as FPnet
from structure.Protonet import ProtoNet as protonet
from structure.PredictionHead import PredictionHead as prednet


class YOLACT(nn.Module):
    def __init__(self, device):
        super(YOLACT, self).__init__()
        self.fpnet = FPnet(device=device)
        self.protonet = protonet()
        self.prednet = prednet()

        self.fpnet.to(device=device)
        self.protonet.to(device=device)
        self.prednet.to(device=device)

        # in test time, calculate semantic segmentation loss.
        # self.conv_seg = torch.nn.Conv2d(256, 1,)

    def forward(self, x):
        p3, p4, p5, p6, p7 = self.fpnet(x)

        # Proto-net and prediction head.
        # # proto types branch.
        proto_types = self.protonet(p3)

        # class : n_class * n_anchor
        # box : 4 * n_anchor
        # mask : n_proto * n_anchor
        map_class_p3, map_box_p3, map_mask_p3 = self.prednet(p3)
        map_class_p4, map_box_p4, map_mask_p4 = self.prednet(p4)
        map_class_p5, map_box_p5, map_mask_p5 = self.prednet(p5)
        map_class_p6, map_box_p6, map_mask_p6 = self.prednet(p6)
        map_class_p7, map_box_p7, map_mask_p7 = self.prednet(p7)

        map_classes = {'p3': map_class_p3,
                       'p4': map_class_p4,
                       'p5': map_class_p5,
                       'p6': map_class_p6,
                       'p7': map_class_p7}
        map_boxes = {'p3': map_box_p3,
                     'p4': map_box_p4,
                     'p5': map_box_p5,
                     'p6': map_box_p6,
                     'p7': map_box_p7}
        map_coef = {'p3': map_mask_p3,
                     'p4': map_mask_p4,
                     'p5': map_mask_p5,
                     'p6': map_mask_p6,
                     'p7': map_mask_p7}

        '''
        proto_types : {N, C, H, W}; C = n_proto(=4)
        map_classes : (each dic){N, C, H, W}; C = n_class(=10) * n_anchor(=3), (H, W) = 64, 32, 16, 8, 4
        map_boxes   : (each dic){N, C, H, W}; C = 4 * n_anchor(=3), (H, W) =  64, 32, 16, 8, 4
        map_coef   : (each dic){N, C, H, W}; C = n_proto(=4) * n_anchor(=3), (H, W) =  64, 32, 16, 8, 4
        '''
        return proto_types, map_classes, map_boxes, map_coef

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        state_dict = torch.load(path)
# # foward test.
# x = torch.ones(1, 3, 512, 512)
# yolacat = YOLACT()
# out = yolacat(x)
# print("finish")
