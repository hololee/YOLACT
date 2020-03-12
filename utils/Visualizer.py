import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class Visualizer(object):
    is_score_showing = False
    is_bbox_showing = False

    def __init__(self, **options: bool):
        global is_score_showing, is_bbox_showing
        is_score_showing = options["score_visibility"]
        is_bbox_showing = options["bbox_visibility"]

    class visualize(object):

        def __init__(self, trigger, img, title, *args):
            '''

            Args:
                valid_boxes: {N_boxes, C}  C : [pr_ch, pr_cw , pr_h, pr_w]
                valid_scores: {N_boxes, }  1 : [class_score]
                valid_coefs: {N_boxes, num_prototype}  1 : [,,.... ,num_prototype]

                mask : {n_box, H, W}
            '''

            # print("show score :", is_score_showing)
            # print("show bbox :", is_bbox_showing)

            self.valid_boxes, self.valid_scores, self.valid_coefs, self.proto_types = args
            # TODO: draw boxes and segment.

            fig1, (ax1, ax2, ax3) = plt.subplots(1, 3)

            # predict boxes.
            num_predict_boxes = len(self.valid_boxes)
            for i in range(num_predict_boxes):

            predict_ch = (pr_l_ch.item() * a_h) + a_ch
            predict_cw = (pr_l_cw.item() * a_w) + a_cw
            predict_h = a_h * np.power(10, pr_l_h.item())
            predict_w = a_w * np.power(10, pr_l_w.item())

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
            ax1.imshow(np.transpose(img[i_batch:i_batch + 1].cpu().numpy(), [0, 2, 3, 1]).squeeze())
            ax1.add_patch(rect2)
            ax1.add_patch(rect1)
            colors = ['red', 'green']
            labels = ['predict bbox', 'G.T bbox']

            # add legend for rect.
            lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in colors]
            plt.legend(lines, labels, bbox_to_anchor=(1, 2))

            # calculate rect. gt and predict.
            # rect1 = patches.Rectangle(
            #     ((predict_cw - predict_w / 2), predict_ch - predict_h / 2),
            #     predict_w, predict_h,
            #     linewidth=1,
            #     edgecolor='r',
            #     facecolor='none')
            # rect2 = patches.Rectangle(
            #     ((gt_cw - gt_w / 2), gt_ch - gt_h / 2),
            #     gt_w, gt_h,
            #     linewidth=1,
            #     edgecolor='g',
            #     facecolor='none')
            ax2.imshow(out.numpy())
            # ax2.add_patch(rect2)
            # ax2.add_patch(rect1)
            # calculate rect. gt and predict.
            # rect1 = patches.Rectangle(
            #     ((predict_cw - predict_w / 2), predict_ch - predict_h / 2),
            #     predict_w, predict_h,
            #     linewidth=1,
            #     edgecolor='r',
            #     facecolor='none')
            # rect2 = patches.Rectangle(
            #     ((gt_cw - gt_w / 2), gt_ch - gt_h / 2),
            #     gt_w, gt_h,
            #     linewidth=1,
            #     edgecolor='g',
            #     facecolor='none')
            ax3.imshow(goal.clone().squeeze().cpu().detach().numpy())

            plt.title(title)

        def show(self):
            plt.show()
