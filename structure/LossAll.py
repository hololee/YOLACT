import torch
import torch.nn as nn
import numpy as np
import utils.config as cfg
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
import imageio
import torchvision
import matplotlib.patches as patches
from matplotlib.lines import Line2D


class AllLoss(nn.Module):

    def __init__(self):
        super(AllLoss, self).__init__()

        '''
          proto_types : {N, C, H, W}; C = n_proto(=4)
          map_classes : (p3 to p7){N, C, H, W}; C = n_class(=1) * n_anchor(=3), (H, W) = 64, 32, 16, 8, 4
          map_boxes   : (p3 to p7){N, C, H, W}; C = 4 * n_anchor(=3), (H, W) =  64, 32, 16, 8, 4
          map_coef   : (p3 to p7){N, C, H, W}; C = n_proto(=4) * n_anchor(=3), (H, W) =  64, 32, 16, 8, 4
          anchor_centers : (p3 to p7){2, H, W}; [0]: coordinate_h, [1] : coordinate_w , (H, W) =  64, 32, 16, 8, 4
          anchor_boxes : (p3 to p7){3, 2};[0:1]:ratio-(1:2),[0:2]:ratio-(1:1),[0:3]:ratio-(2:1) [1:0]: height of box, [1:1] : width of box , * area = [24, 48, 96, 192, 384] squared in paper
          anchor_IOUs = (p3 to p7){N, C, H, W}; C = n_anchor(=3), (H, W) =  64, 32, 16, 8, 4  * anchor ratio = [1, 1/2, 2]
          anchor_classes = (p3 to p7){N, C, H, W} C =  n_anchor(=3) * n_class(=1), (H, W) =  64, 32, 16, 8, 4  * anchor ratio = [1, 1/2, 2]
          anchor_gt_idxs = (p3 to p7){N, C, H, W}; C = n_anchor(=3), (H, W) =  64, 32, 16, 8, 4  * anchor ratio = [1, 1/2, 2] : index of gt.box
          '''

        self.crit_cls = torch.nn.BCELoss(reduction='sum')
        self.crit_loc = torch.nn.SmoothL1Loss(reduction='sum')
        self.crit_mask = torch.nn.BCELoss(reduction='mean')

    def forward(self, device, model, img, target, anchor_handler, plot=False):
        print("calculate loss")

        proto_types, map_classes, map_boxes, map_coefs = model(img)
        anchor_centers, anchor_boxes = anchor_handler.anchor_centers, anchor_handler.anchor_boxes
        anchor_IOUs, anchor_classes, anchor_gt_idxs = anchor_handler.calculate_whole_iou(target)

        num_matched_anchors = 0
        loss_cls_positive = 0
        loss_cls_negative = 0
        loss_localization = 0
        loss_mask = 0

        for p_layer in anchor_handler.layers:  # 'p3', 'p4' .... 'p7'

            anchor_center = anchor_centers[p_layer]
            anchor_box = anchor_boxes[p_layer]
            anchor_gt_idx = anchor_gt_idxs[p_layer]
            anchor_class = anchor_classes[p_layer]

            map_class = map_classes[p_layer]
            map_box = map_boxes[p_layer]
            map_coef = map_coefs[p_layer]

            num_matched_anchors += len(np.nonzero(anchor_class)[0])
            cur_num_anchors = len(np.nonzero(anchor_class)[0])

            if cur_num_anchors > 0:

                # # calculate cls loss
                output_class_pos = map_class[np.nonzero(anchor_class)]
                output_class_pos.view(-1, cfg.num_classes)
                output_class_neg = map_class[np.where(anchor_class == 0)][:3 * len(output_class_pos)]
                output_class_neg.view(-1, cfg.num_classes)

                output_class_pos = torch.exp(output_class_pos) / (torch.exp(output_class_pos) + torch.exp(torch.tensor(0.)))
                output_class_neg = torch.exp(output_class_neg) / (torch.exp(output_class_neg) + torch.exp(torch.tensor(0.)))

                # TODO: calculate loss and train network.
                loss_cls_positive += self.crit_cls(output_class_pos,
                                                   torch.from_numpy(np.ones(shape=output_class_pos.shape, dtype=np.float32)).to(device))
                loss_cls_negative += self.crit_cls(output_class_neg,
                                                   torch.from_numpy(np.zeros(shape=output_class_neg.shape, dtype=np.float32)).to(device))

                # # calculate localization loss.
                for i in range(cur_num_anchors):
                    # TODO : CHECK AGAIN. IF HAVE MULTIPLE CLASSES.  (anchor_class)
                    anchor_coord = np.array([np.nonzero(anchor_class)[2][i], np.nonzero(anchor_class)[3][i]])  # h, w
                    anchor_ratio = np.nonzero(anchor_class)[1][i]
                    gt_type = anchor_gt_idx[0, anchor_ratio, anchor_coord[0], anchor_coord[1]]

                    a_ch, a_cw, a_h, a_w = anchor_center[0, anchor_coord[0], anchor_coord[1]], \
                                           anchor_center[1, anchor_coord[0], anchor_coord[1]], \
                                           anchor_box[anchor_ratio, 0], anchor_box[anchor_ratio, 1]
                    gt_ch, gt_cw, gt_h, gt_w = [target["boxes"][0, int(gt_type), x] for x in range(4)]
                    pr_l_ch, pr_l_cw, pr_l_h, pr_l_w = [map_box[0, anchor_ratio * 4 + x, anchor_coord[0], anchor_coord[1]] for x in range(4)]

                    loss_localization += self.crit_loc(pr_l_ch.view(1, -1), ((gt_ch - a_ch) / a_h).view(1, -1))  # add loss ch
                    loss_localization += self.crit_loc(pr_l_cw.view(1, -1), ((gt_cw - a_cw) / a_w).view(1, -1))  # add loss cw
                    loss_localization += self.crit_loc(pr_l_h.view(1, -1), torch.log10(gt_h / a_h).view(1, -1))  # add loss h
                    loss_localization += self.crit_loc(pr_l_w.view(1, -1), torch.log10(gt_w / a_w).view(1, -1))  # add loss w

                    # calculating mask coefficients.
                    co_ef = torch.stack(([map_coef[0, anchor_ratio * 4 + x, anchor_coord[0], anchor_coord[1]] for x in range(4)]))
                    proto_type = torch.stack([proto_types[:, x, :, :] for x in range(4)], dim=1).squeeze()

                    mul = proto_type.squeeze() * co_ef.squeeze().unsqueeze(dim=-1).unsqueeze(dim=-1)
                    mask_result = torch.sigmoid(mul.sum(0).unsqueeze(0))
                    predict = F.interpolate(mask_result.unsqueeze(0), size=512)
                    goal = target["masks"][:, int(gt_type), :, :].unsqueeze(0)

                    # plot prediction.
                    if plot:
                        predict_ch = (pr_l_ch.item() * a_h) + a_ch
                        predict_cw = (pr_l_cw.item() * a_w) + a_cw
                        predict_h = a_h * np.power(10, pr_l_h.item())
                        predict_w = a_w * np.power(10, pr_l_w.item())

                        fig1, (ax1, ax2, ax3) = plt.subplots(1, 3)
                        # calculate rect. gt and predict.
                        rect1 = patches.Rectangle(
                            ((predict_cw - predict_w / 2), predict_ch - predict_h / 2),
                            predict_w, predict_h,
                            linewidth=1,
                            edgecolor='r',
                            facecolor='none')
                        rect2 = patches.Rectangle(
                            ((gt_cw - gt_w / 2), gt_ch - gt_h / 2),
                            gt_w, gt_h,
                            linewidth=1,
                            edgecolor='g',
                            facecolor='none')
                        ax1.imshow(np.transpose(img.cpu().numpy(), [0, 2, 3, 1]).squeeze())
                        ax1.add_patch(rect2)
                        ax1.add_patch(rect1)
                        colors = ['red', 'green']
                        labels = ['predict bbox', 'G.T bbox']

                        # add legend for rect.
                        lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in colors]
                        plt.legend(lines, labels, bbox_to_anchor=(1, 2))

                        # calculate rect. gt and predict.
                        rect1 = patches.Rectangle(
                            ((predict_cw - predict_w / 2), predict_ch - predict_h / 2),
                            predict_w, predict_h,
                            linewidth=1,
                            edgecolor='r',
                            facecolor='none')
                        rect2 = patches.Rectangle(
                            ((gt_cw - gt_w / 2), gt_ch - gt_h / 2),
                            gt_w, gt_h,
                            linewidth=1,
                            edgecolor='g',
                            facecolor='none')
                        ax2.imshow(predict.clone().squeeze().cpu().detach().numpy())
                        ax2.add_patch(rect2)
                        ax2.add_patch(rect1)
                        # calculate rect. gt and predict.
                        rect1 = patches.Rectangle(
                            ((predict_cw - predict_w / 2), predict_ch - predict_h / 2),
                            predict_w, predict_h,
                            linewidth=1,
                            edgecolor='r',
                            facecolor='none')
                        rect2 = patches.Rectangle(
                            ((gt_cw - gt_w / 2), gt_ch - gt_h / 2),
                            gt_w, gt_h,
                            linewidth=1,
                            edgecolor='g',
                            facecolor='none')
                        ax3.imshow(goal.clone().squeeze().cpu().detach().numpy())
                        ax3.add_patch(rect2)
                        ax3.add_patch(rect1)
                        # plt.show()
                        plt.savefig('/home/user01/data_ssd/LeeJongHyeok/pytorch_project/YOLACT/image/{}.png'.format(time.time()))
                        # imageio.imwrite('/home/user01/data_ssd/LeeJongHyeok/pytorch_project/YOLACT/image/{}.png'.format(time.time()),
                        #                 np.transpose(img.cpu().numpy(), [0, 2, 3, 1]).squeeze().astype(np.uint8))
                        # imageio.imwrite('/home/user01/data_ssd/LeeJongHyeok/pytorch_project/YOLACT/image/{}.png'.format(time.time()),
                        #                 predict.clone().squeeze().cpu().detach().numpy().astype(np.uint8))
                        # imageio.imwrite('/home/user01/data_ssd/LeeJongHyeok/pytorch_project/YOLACT/image/{}.png'.format(time.time()),
                        #                 goal.clone().squeeze().cpu().detach().numpy())
                    loss_mask += self.crit_mask(predict, goal)
                else:
                    pass

        if num_matched_anchors > 0:
            total_loss = ((loss_cls_positive + loss_cls_negative + cfg.loss_cls_alpha * loss_localization) + loss_mask / num_matched_anchors)
            return total_loss
        else:
            print("no matched anchor box.")
            return torch.tensor(0)


"""SSD Weighted Loss Function
   Compute Targets:
       1) Produce Confidence Target Indices by matching  ground truth boxes
          with (default) 'priorboxes' that have jaccard index > threshold parameter
          (default threshold: 0.5).
       2) Produce localization target by 'encoding' variance into offsets of ground
          truth boxes and their matched  'priorboxes'.
       3) Hard negative mining to filter the excessive number of negative examples
          that comes with using a large number of default bounding boxes.
          (default negative:positive ratio 3:1)
   Objective Loss:
       L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
       Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
       weighted by α which is set to 1 by cross val.
       Args:
           c: class confidences,
           l: predicted boxes,
           g: ground truth boxes
           N: number of matched default boxes
       See: https://arxiv.org/pdf/1512.02325.pdf for more details.
   """
