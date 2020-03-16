import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.patches as patches
import utils.config as cfg


class Visualizer(object):
    is_score_showing = False
    is_bbox_showing = False

    def __init__(self, **options: bool):
        global is_score_showing, is_bbox_showing
        is_score_showing = options["score_visibility"]
        is_bbox_showing = options["bbox_visibility"]

    class visualize(object):

        def __init__(self, img, title, *args):
            '''

            Args:
                img : (3, 512, 512)
                valid_boxes: {N_boxes, C}  C : [pr_ch, pr_cw , pr_h, pr_w]
                valid_scores: {N_boxes, }  1 : [class_score]
                valid_coefs: {N_boxes, num_prototype}  1 : [,,.... ,num_prototype]
                anchor_centers : (N_boxes, 2) : ch, cw
                anchor_boxes : (N_boxes, 2) : h, w

               target_boxes = {n_box, 4}
               target_masks = { n_box, H, W}
            '''

            # print("show score :", is_score_showing)
            # print("show bbox :", is_bbox_showing)

            color_palette = ((255, 149, 132),
                             (149, 85, 66),
                             (250, 202, 87),
                             (131, 209, 96),
                             (40, 202, 204),
                             (36, 137, 176),
                             (221, 160, 246),
                             (247, 72, 119),
                             (0, 173, 95),
                             (255, 75, 90),
                             (0, 187, 236),
                             (189, 89, 212),
                             (255, 131, 0),
                             (235, 72, 71),
                             (128, 88, 189),
                             (0, 181, 233),
                             (98, 216, 182),
                             (194, 226, 84),
                             (255, 217, 101),
                             (180, 91, 62))

            valid_boxes, valid_scores, valid_coefs, proto_types, target_boxes, target_masks = args

            # draw predict boxes.
            self.num_predict_boxes = len(valid_boxes)
            self.rects = []
            self.scores = []
            self.mask_predict = np.zeros([3, 128, 128], dtype=np.int)
            self.mask_target = np.zeros([3, 512, 512], dtype=np.int)

            if self.num_predict_boxes > len(color_palette):
                print("number of boxes over the color palette.")
                self.num_predict_boxes = len(color_palette)

            for i in range(self.num_predict_boxes):
                predict_ch, predict_cw, predict_h, predict_w = valid_boxes[i]




                # calculate rect. gt and predict.
                rect = patches.Rectangle(
                    ((predict_cw - predict_w / 2), predict_ch - predict_h / 2),
                    predict_w, predict_h,
                    linewidth=1,
                    edgecolor='r',
                    facecolor='none')

                self.rects.append(rect)
                self.scores.append(valid_scores[i])

                # calculate segmentation image.
                mul = proto_types.clone().cpu().detach().numpy().squeeze().transpose([1, 2, 0]) * valid_coefs[i]
                out = 1 / (1 + np.exp(-mul.sum(2)))  # output segmentation.

                coord_x_y = np.where(out > cfg.pred_th)

                self.mask_predict[0, coord_x_y[0], coord_x_y[1]] = color_palette[i][0]
                self.mask_predict[1, coord_x_y[0], coord_x_y[1]] = color_palette[i][1]
                self.mask_predict[2, coord_x_y[0], coord_x_y[1]] = color_palette[i][2]

            for i, mask_image in enumerate(target_masks.cpu().detach().numpy()):
                coord_x_y = np.where(mask_image > 0)

                self.mask_target[0, coord_x_y[0], coord_x_y[1]] = color_palette[i][0]
                self.mask_target[1, coord_x_y[0], coord_x_y[1]] = color_palette[i][1]
                self.mask_target[2, coord_x_y[0], coord_x_y[1]] = color_palette[i][2]

            self.origin = np.transpose(img.clone().cpu().detach().numpy(), [1, 2, 0]).squeeze()
            self.mask_predict = np.transpose(self.mask_predict, [1, 2, 0])
            self.mask_target = np.transpose(self.mask_target, [1, 2, 0])

            # skip target box.

            '''
            self.rects
            self.scores
            self.mask_predict
            self.mask_target
            '''

        def show(self, path, is_show=False):
            # notice: draw boxes and segment.
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax1.imshow(self.origin)

            for idx in range(len(self.rects)):
                ax1.add_patch(self.rects[idx])
                x, y = self.rects[idx].get_xy()
                ax1.annotate(str(self.scores[0])[:4], (x, y), color='w', weight='bold', fontsize=8, ha='left', va='top')

            ax2.imshow(self.mask_predict)
            ax3.imshow(self.mask_target)

            ax1.set_title("predict boxes : {}".format(self.num_predict_boxes))
            ax2.set_title("predict seg")
            ax3.set_title("target seg")

            plt.savefig(path)
            if is_show:
                plt.show()
