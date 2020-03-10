import torch
import utils.config as cfg
import numpy as np
import matplotlib
from operator import add, sub

# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time


# create anchor center coordinates.
# create anchor regions.
# save the index of object and background.
class AnchorHandler:
    def __init__(self, img_h, img_w):

        '''
        img      : {N, 3, H, W}
        proto_types : {N, C, H, W}; C = n_proto(=4)
        map_classes : (p3 to p7){N, C, H, W}; C = n_class(=10) * n_anchor(=3), (H, W) = 64, 32, 16, 8, 4
        map_boxes   : (p3 to p7){N, C, H, W}; C = 4 * n_anchor(=3), (H, W) =  64, 32, 16, 8, 4  * anchor ratio = [1, 1/2, 2]
        map_masks   : (p3 to p7){N, C, H, W}; C = n_proto(=4) * n_anchor(=3), (H, W) =  64, 32, 16, 8, 4

        target : {"boxes": boxes, "labels": labels, "masks": masks}

        boxes  : ([N, num_boxes, (ch, cw, h, w))])
        '''

        self.img_h = img_h
        self.img_w = img_w

        self.layers_T = ['p3', 'p4', 'p5', 'p6', 'p7']
        self.squared_area = {'p3': 24, 'p4': 48, 'p5': 96, 'p6': 192, 'p7': 384}
        self.feature_height = {'p3': img_h // (2 ** 3), 'p4': img_h // (2 ** 4), 'p5': img_h // (2 ** 5),
                               'p6': img_h // (2 ** 6), 'p7': img_h // (2 ** 7)}
        self.feature_width = {'p3': img_w // (2 ** 3), 'p4': img_w // (2 ** 4), 'p5': img_w // (2 ** 5),
                              'p6': img_w // (2 ** 6), 'p7': img_w // (2 ** 7)}

        self.layers = self.layers_T[cfg.using_start_layer_no: cfg.using_stop_layer_no]

        self.anchor_centers = {}
        self.anchor_boxes = {}

    def cal_anchor(self):
        print("--Calculate anchor center--")
        for px in self.layers:
            # print("calculate anchor : layer", px)
            anchor_center = np.zeros([2, self.feature_height[px], self.feature_width[px]])

            h_count = self.feature_height[px]
            w_count = self.feature_width[px]

            # standard_anchor : origin size of one pixel in p_layer.
            standard_anchor_h = self.img_h / h_count
            standard_anchor_w = self.img_w / w_count

            for c_h in range(h_count):
                for c_w in range(w_count):
                    # calculate centers.
                    ctr_h = (c_h * standard_anchor_h) + (standard_anchor_h - 1) / 2
                    ctr_w = (c_w * standard_anchor_w) + (standard_anchor_w - 1) / 2

                    anchor_center[0, c_h, c_w], anchor_center[1, c_h, c_w] = ctr_h, ctr_w
                    # print(ctr_h, ctr_w)

            self.anchor_centers[px] = anchor_center

        print("--Calculate anchor H, W--")
        for px in self.layers:
            # print("calculate anchor : layer", px)

            anchor_box = np.zeros([cfg.a, 2])

            for ratio in range(cfg.a):
                '''
                0 : 1:2
                1 : 1:1
                2 : 2:1
                '''
                # init
                box_h, box_w = 0, 0

                area = self.squared_area[px]
                if ratio == 0:
                    box_h = 1 * area * (1 / np.sqrt(2))
                    box_w = 2 * area * (1 / np.sqrt(2))
                elif ratio == 1:
                    box_h = area
                    box_w = area
                elif ratio == 2:
                    box_h = 2 * area * (1 / np.sqrt(2))
                    box_w = 1 * area * (1 / np.sqrt(2))

                anchor_box[ratio, 0], anchor_box[ratio, 1] = box_h, box_w

            self.anchor_boxes[px] = anchor_box

        '''
        anchor_centers : (p3 to p7){2, H, W}; [0]: coordinate_h, [1] : coordinate_h_w , (H, W) =  64, 32, 16, 8, 4
        anchor_boxes : (p3 to p7){3, 2};[0:1]:ratio-(1:2),[0:2]:ratio-(1:1),[0:3]:ratio-(2:1) [1:0]: height of box, [1:1] : width of box , * area = [24, 48, 96, 192, 384] squared in paper
        '''
        # return self.anchor_centers, self.anchor_boxes

    def calculate_iou_one(self, box1, box2):
        '''
        :param box1: (tuple) center_h, center_w , box_h, box_w
        :param box2: (tuple) center_h, center_w , box_h, box_w
        :return: iou value
        '''
        box1_LT = [box1[0] - box1[2] / 2, box1[1] - box1[3] / 2]
        box1_RB = [box1[0] + box1[2] / 2, box1[1] + box1[3] / 2]

        box2_LT = [box2[0] - box2[2] / 2, box2[1] - box2[3] / 2]
        box2_RB = [box2[0] + box2[2] / 2, box2[1] + box2[3] / 2]

        box1_LT = np.clip(box1_LT, 0, self.img_h)
        box1_RB = np.clip(box1_RB, 0, self.img_h)

        box2_LT = np.clip(box2_LT, 0, self.img_h)
        box2_RB = np.clip(box2_RB, 0, self.img_h)

        # rectangle nonzero check.
        assert box1_LT[1] < box1_RB[1], "box1 width is negative"
        assert box1_LT[0] < box1_RB[0], "box1 height is negative"
        assert box2_LT[1] < box2_RB[1], "box2 width is negative"
        assert box2_LT[0] < box2_RB[0], "box2 height is negative"

        # determine the coordinates of the intersection rectangle
        x_left = max(box1_LT[1], box2_LT[1])
        y_top = max(box1_LT[0], box2_LT[0])
        x_right = min(box1_RB[1], box2_RB[1])
        y_bottom = min(box1_RB[0], box2_RB[0])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (box1_RB[1] - box1_LT[1]) * (box1_RB[0] - box1_LT[0])
        bb2_area = (box2_RB[1] - box2_LT[1]) * (box2_RB[0] - box2_LT[0])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0

        # print("iou :", iou)

        return iou.cpu().numpy() if isinstance(iou, torch.Tensor) else iou

    def calculate_iou_matrix(self, *args):
        x11, y11, x12, y12 = args[:4]
        x21, y21, x22, y22 = args[4:]
        xA = np.maximum(x11, np.transpose(x21))
        yA = np.maximum(y11, np.transpose(y21))
        xB = np.minimum(x12, np.transpose(x22))
        yB = np.minimum(y12, np.transpose(y22))
        interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
        boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
        boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
        iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
        return iou

    def calculate_whole_iou(self, target):

        anchor_gt_idxs = {}
        anchor_IOUs = {}
        anchor_classes = {}

        # print("--Calculate anchor IOU of Bbox--")
        for px in self.layers:
            # print("calculate anchor : layer", px)

            anchor_gt_idx = np.zeros([1, cfg.a, self.feature_height[px], self.feature_width[px]])
            anchor_IOU = np.zeros([1, cfg.a, self.feature_height[px], self.feature_width[px]])
            anchor_class = np.zeros([1, cfg.a * cfg.num_classes, self.feature_height[px], self.feature_width[px]])

            # calculate each batch
            # {C, H, W}; C = 4 * n_anchor(=3), (H, W) =  64, 32, 16, 8, 4  * anchor ratio = [1, 1/2, 2]
            # ratios = {'1': map_box[:4], '1/2': map_box[:4:8], '2': map_box[8:]}

            h_count = self.feature_height[px]
            w_count = self.feature_width[px]

            # for ratio in range(cfg.a):
            #
            #     # # index's anchor box information.
            #     anchor_ctr_h = [self.anchor_centers[px][0, c_h, c_w] for c_h in range(h_count) for c_w in range(w_count)]
            #     anchor_ctr_w = [self.anchor_centers[px][1, c_h, c_w] for c_h in range(h_count) for c_w in range(w_count)]
            #     anchor_box_h = [self.anchor_boxes[px][ratio, 0] for c_h in range(h_count) for c_w in range(w_count)]
            #     anchor_box_w = [self.anchor_boxes[px][ratio, 1] for c_h in range(h_count) for c_w in range(w_count)]
            #
            #     anchor_count = len(anchor_ctr_h)
            #
            #     gt_ious = []
            #
            #     for gt_box in range(target["labels"].size()[1]):
            #         # label box : {ch ,cw, h, w}
            #         label_ctr_h = target["boxes"][0, gt_box, 0].cpu().numpy()
            #         label_ctr_w = target["boxes"][0, gt_box, 1].cpu().numpy()
            #         label_box_h = target["boxes"][0, gt_box, 2].cpu().numpy()
            #         label_box_w = target["boxes"][0, gt_box, 3].cpu().numpy()
            #
            #         boxes = np.array([[anchor_ctr_w[i] - anchor_box_w[i] / 2,
            #                            anchor_ctr_h[i] - anchor_box_h[i] / 2,
            #                            anchor_ctr_w[i] + anchor_box_w[i] / 2,
            #                            anchor_ctr_h[i] + anchor_box_h[i] / 2,
            #                            label_ctr_w - label_box_w / 2,
            #                            label_ctr_h - label_box_h / 2,
            #                            label_ctr_w + label_box_w / 2,
            #                            label_ctr_h + label_box_h / 2]
            #                           for i in range(anchor_count)])
            #
            #         ious = self.calculate_iou_matrix(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4], boxes[:, 5], boxes[:, 6],
            #                                          boxes[:, 7])  # (0,0) (0,1) (0,2) ... (N,N)
            #
            #         gt_ious.append(ious)  # different g.ts at same anchor.
            #
            #     gt_ious = np.array(gt_ious)
            #     max_iou = [np.max(gt_ious[:, i]) for i in range(len(gt_ious[0]))]
            #     max_iou_index = [np.argmax(gt_ious[:, i]) for i in range(len(gt_ious[0]))]
            #
            #     for index, iou in enumerate(max_iou):
            #         if iou >= cfg.default_detect_iou:  # if the maximum iou >= 0.5.
            #             anchor_IOU[0, ratio, int(index / h_count), int(index % h_count)] = iou
            #             anchor_gt_idx[0, ratio, int(index / h_count), int(index % h_count)] = int(max_iou_index[index])
            #             anchor_class[0, ratio * cfg.num_classes, int(index / h_count), int(index % h_count)] = 1
            #
            # anchor_gt_idxs[px] = anchor_gt_idx
            # anchor_IOUs[px] = anchor_IOU
            # anchor_classes[px] = anchor_class

            # # index's anchor box information.
            anchor_ctr_h = [self.anchor_centers[px][0, c_h, c_w] for ratio in range(cfg.a) for c_h in range(h_count) for c_w in range(w_count)]
            anchor_ctr_w = [self.anchor_centers[px][1, c_h, c_w] for ratio in range(cfg.a) for c_h in range(h_count) for c_w in range(w_count)]
            anchor_box_h = [self.anchor_boxes[px][ratio, 0] for ratio in range(cfg.a) for c_h in range(h_count) for c_w in range(w_count)]
            anchor_box_w = [self.anchor_boxes[px][ratio, 1] for ratio in range(cfg.a) for c_h in range(h_count) for c_w in range(w_count)]

            anchor_count = len(anchor_ctr_h)

            gt_ious = []

            for gt_box in range(target["boxes"].size()[1]):
                # label box : {ch ,cw, h, w}
                label_ctr_h = target["boxes"][0, gt_box, 0].cpu().numpy()
                label_ctr_w = target["boxes"][0, gt_box, 1].cpu().numpy()
                label_box_h = target["boxes"][0, gt_box, 2].cpu().numpy()
                label_box_w = target["boxes"][0, gt_box, 3].cpu().numpy()

                boxes = np.array([[anchor_ctr_w[i] - anchor_box_w[i] / 2,
                                   anchor_ctr_h[i] - anchor_box_h[i] / 2,
                                   anchor_ctr_w[i] + anchor_box_w[i] / 2,
                                   anchor_ctr_h[i] + anchor_box_h[i] / 2,
                                   label_ctr_w - label_box_w / 2,
                                   label_ctr_h - label_box_h / 2,
                                   label_ctr_w + label_box_w / 2,
                                   label_ctr_h + label_box_h / 2]
                                  for i in range(anchor_count)])

                # all location iou with specific gt_box.
                ious = self.calculate_iou_matrix(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4], boxes[:, 5], boxes[:, 6],
                                                 boxes[:, 7])  # (0,0) (0,1) (0,2) ... (N,N)

                gt_ious.append(ious)  # different g.ts at same anchor.

            gt_ious = np.array(gt_ious)
            max_iou = [np.max(gt_ious[:, i]) for i in range(len(gt_ious[0]))]
            max_iou_index = [np.argmax(gt_ious[:, i]) for i in range(len(gt_ious[0]))]

            for index, iou in enumerate(max_iou):
                ratio = index // (h_count * w_count)
                offset = (ratio * (h_count * w_count))

                if iou >= cfg.default_detect_iou:  # if the maximum iou >= 0.5.
                    anchor_IOU[0, ratio, int((index - offset) // h_count), int((index - offset) % h_count)] = iou
                    anchor_gt_idx[0, ratio, int((index - offset) // h_count), int((index - offset) % h_count)] = int(max_iou_index[index])
                    # TODO : CHECK AGAIN. IF HAVE MULTIPLE CLASSES.
                    anchor_class[0, ratio * cfg.num_classes, int((index - offset) / h_count), int((index - offset) % h_count)] = 1

            anchor_gt_idxs[px] = anchor_gt_idx
            anchor_IOUs[px] = anchor_IOU
            anchor_classes[px] = anchor_class

        '''
        anchor_IOUs = (p3 to p7){1, C, H, W}; C = n_anchor(=3), (H, W) =  64, 32, 16, 8, 4  * anchor ratio = [1, 1/2, 2]
        anchor_classes = (p3 to p7){1, C, H, W} C =  n_anchor(=3) *  n_class(=1), (H, W) =  64, 32, 16, 8, 4  * anchor ratio = [1, 1/2, 2]
        '''
        return anchor_IOUs, anchor_classes, anchor_gt_idxs


'''
fig, ax = plt.subplots(1)
ax.imshow(np.transpose(img, [0, 2, 3, 1]).squeeze())
rect1 = patches.Rectangle(
    (512 - (anchor_ctr_w - anchor_box_w / 2), anchor_ctr_h - anchor_box_h / 2),
    anchor_box_w, anchor_box_h,
    linewidth=1,
    edgecolor='r',
    facecolor='none')
rect2 = patches.Rectangle(
    (512 - (label_ctr_w - label_box_w / 2), label_ctr_h - label_box_h / 2),
    label_box_w, label_box_h,
    linewidth=1,
    edgecolor='g',
    facecolor='none')

ax.add_patch(rect2)
ax.add_patch(rect1)

plt.show()
'''
