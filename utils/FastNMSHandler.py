import torch


class FastNMS(object):
    def __init__(self, boxes):
        '''

        Args:
            boxes: ({N_boxes, 9}, {} ...{n}) (tuple)  n = num_batches  9 : [an_ch, an_cw ,an_h, an_w , pr_ch, pr_cw , pr_h, pr_w, class_score]
        '''

        self.num_batches = len(boxes)
        self.num_boxes = [boxes[i_batch].size()[0] for i_batch in range(self.num_batches)]  # same as batch size.

    def __call__(self):
        return 0.
