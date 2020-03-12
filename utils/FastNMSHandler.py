import numpy as np
from utils.AnchorHandler import AnchorHandler
import utils.config as cfg
from itertools import permutations, product


class FastNMS(object):
    def __init__(self):
        self.ah = AnchorHandler(cfg.image_height, cfg.image_width)
        pass

    def __call__(self, boxes, scores, coefs):
        '''

        Args:
           boxes: {N_boxes, C}  C : [pr_ch, pr_cw , pr_h, pr_w]
           scores: {N_boxes, 1}  1 : [class_score]
           coefs:   {N_boxes, N_prototypes}
        Returns: valid matrix.

        '''

        cls_scores_sorted = np.sort(scores[:, 0])  # ascending.
        cls_scores_indices = np.argsort(scores[:, 0])  # ascending.
        boxes_sorted = boxes[cls_scores_indices]
        coefs_sorted = coefs[cls_scores_indices]

        # change to descending. (reverse array)
        self.cls_scores_sorted = cls_scores_sorted[::-1]
        self.boxes_sorted = boxes_sorted[::-1]
        self.coefs_sorted = coefs_sorted[::-1]

        # choose top k scores.
        self.cls_scores_sorted = self.cls_scores_sorted[:min(cfg.top_k, len(self.cls_scores_sorted))]
        self.boxes_sorted = self.boxes_sorted[:min(cfg.top_k, len(self.cls_scores_sorted))]
        self.coefs_sorted = self.coefs_sorted[:min(cfg.top_k, len(self.cls_scores_sorted))]

        # product_list = np.reshape(list(product(list_boxes, repeat=2)), (cfg.top_k, cfg.top_k, 9))

        # descendig list. ex) (4, 4) , (4 ,3)  ...
        ch_matrix, cw_matrix, h_matrix, w_matrix = [np.array(list(product(self.boxes_sorted[:, i], repeat=2))).squeeze() for i in range(4)]
        # AnchorHandler.calculate_iou_matrix(x11, y11, x12, y12 , x21, y21, x22, y22) #jaccard

        boxes = np.array([[cw_matrix[i, 0] - w_matrix[i, 0] / 2,
                           ch_matrix[i, 0] - h_matrix[i, 0] / 2,
                           cw_matrix[i, 0] + w_matrix[i, 0] / 2,
                           ch_matrix[i, 0] + h_matrix[i, 0] / 2,
                           cw_matrix[i, 1] - w_matrix[i, 1] / 2,
                           ch_matrix[i, 1] - h_matrix[i, 1] / 2,
                           cw_matrix[i, 1] + w_matrix[i, 1] / 2,
                           ch_matrix[i, 1] + h_matrix[i, 1] / 2]
                          for i in range(len(ch_matrix))])

        ious = self.ah.calculate_iou_matrix(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4], boxes[:, 5], boxes[:, 6], boxes[:, 7])

        matirx_pairwise_iou_X = np.reshape(ious, (cfg.top_k, cfg.top_k))
        removed_matrix_X = np.triu(matirx_pairwise_iou_X, k=1)

        matrix_K = np.max(removed_matrix_X, axis=0)  # what axis ?
        overlaped_indices = np.where(matrix_K > cfg.fnms_th)  # overlaped indices.
        all_indices = list(range(len(matrix_K)))

        valid_boxes = np.delete(self.boxes_sorted, overlaped_indices, axis=0)
        valid_scores = np.delete(self.cls_scores_sorted, overlaped_indices, axis=0)
        valid_coefs = np.delete(self.coefs_sorted, overlaped_indices, axis=0)
        # valid_indices = np.delete(all_indices, overlaped_indices)

        return valid_boxes, valid_scores, valid_coefs
