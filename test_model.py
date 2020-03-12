import torch
import data.DataHandler as data_handle
import utils.config as cfg
from structure.yolacat import YOLACT as yolact
from utils.AnchorHandler import AnchorHandler as anchor_handle
from utils.FastNMSHandler import FastNMS
from utils.Visualizer import Visualizer
import numpy as np
import time

data_loader_train, data_loader_test = data_handle.get_data(cfg.dataset)

mFastNMS = FastNMS()
mVisualizer = Visualizer(score_visibility=False, bbox_visibility=False)

# notice: load weights.
model = yolact(cfg.device)
model.load_weights("/home/user01/data_ssd/LeeJongHyeok/pytorch_project/YOLACT/model/model_epoch_200_lr_1e-4_bat_20_layer_2to5.pth")
model.to(cfg.device)

# test
model.eval()

anchor_handler = anchor_handle(img_h=cfg.image_height, img_w=cfg.image_width)
anchor_handler.cal_anchor()

with torch.no_grad():
    for iter, (img, target) in enumerate(data_loader_test):
        '''
        img = {N, C, H, W}
        
        target["labels"] = {1, n_box}
        target["boxes"] = {1, n_box, 4}
        target["masks"] = {1, n_box, H, W}
        '''

        img = img.to(cfg.device)
        target["labels"] = target["labels"].to(cfg.device)
        target["boxes"] = target["boxes"].to(cfg.device)
        target["masks"] = target["masks"].to(cfg.device)

        # predict.
        '''
        proto_types : {1, C, H, W}; C = n_proto(=4)
        map_classes : (p3 to p7){1, C, H, W}; C = n_class(=1) * n_anchor(=3), (H, W) = 64, 32, 16, 8, 4
        map_boxes   : (p3 to p7){1, C, H, W}; C = 4 * n_anchor(=3), (H, W) =  64, 32, 16, 8, 4
        map_coef   : (p3 to p7){1, C, H, W}; C = n_proto(=4) * n_anchor(=3), (H, W) =  64, 32, 16, 8, 4
        '''

        start_pr = time.time()

        proto_types, map_classes, map_boxes, map_coefs = model(img)

        # whole anchor located boxes are listed here.
        all_boxes = []
        all_scores = []
        all_coefs = []

        for p_layer in anchor_handler.layers:
            map_class = map_classes[p_layer]
            map_box = map_boxes[p_layer]
            map_coef = map_coefs[p_layer]

            size_h = map_classes[p_layer].size()[2]
            size_w = map_classes[p_layer].size()[3]

            # specific layers sequential boxes. index: a_id *(size_h * size_w) + h_id * (size_w) + w_id
            local_classes = [map_class[0, id_anchor, id_h, id_w].clone().squeeze().cpu().detach().numpy() for id_anchor in range(cfg.a) for id_h in
                             range(size_h) for id_w in range(size_w)]
            local_bboxes = [map_box[0, (id_anchor * 4):(id_anchor * 4) + 4, id_h, id_w].clone().squeeze().cpu().detach().numpy() for id_anchor in range(cfg.a)
                            for id_h in range(size_h) for id_w in
                            range(size_w)]
            local_coefes = [map_coef[0, (id_anchor * cfg.num_prototype):(id_anchor * cfg.num_prototype) + cfg.num_prototype, id_h,
                            id_w].clone().squeeze().cpu().detach().numpy() for id_anchor in
                            range(cfg.a) for id_h in range(size_h) for id_w in range(size_w)]

            all_scores += local_classes  # (n_boxes, )
            all_boxes += local_bboxes  # (n_boxes, 4)
            all_coefs += local_coefes  # (n_boxes, n_prototypes)

        # change to numpy.
        all_scores = np.array(all_scores).reshape([-1, 1])
        all_boxes = np.array(all_boxes)
        all_coefs = np.array(all_coefs)

        # calculate valid box.
        valid_boxes, valid_scores, valid_coefs = mFastNMS(boxes=all_boxes, scores=all_scores, coefs=all_coefs)

        # visualize the image.
        mVisualizer.visualize(mVisualizer.predict, img, "predict", valid_boxes, valid_scores, valid_coefs, proto_types).show()
        mVisualizer.visualize(mVisualizer.target, img, "target", target["boxes"][0], np.ones(shape=np.shape(target["labels"][0])), target["masks"][0]).show()

        take_time = time.time() - start_pr
        print("predict_time :", take_time, "(s)")

        # print("[{}/{}][{}/{}] train loss : {}, TIME : {}s".format(epoch + 1, cfg.total_epoch, iter + 1, len(data_loader_train), total_loss, time.time() - cur))
        # print("EPOCH TIME : {}s".format(time.time() - epoch_time))
