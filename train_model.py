import torch
import data.DataHandler as data_handle
import utils.config as cfg
from structure.yolacat import YOLACT as yolact
from structure.LossAll import AllLoss
from utils.AnchorHandler import AnchorHandler as anchor_handle
import time

data_loader_train, data_loader_test = data_handle.get_data(cfg.dataset)

model = yolact(cfg.device)
model.to(cfg.device)

# train
model.train()

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=0.1)

loss = AllLoss()
anchor_handler = anchor_handle(img_h=cfg.image_height, img_w=cfg.image_width)
anchor_handler.cal_anchor()


def train(device, optimizer, lr_scheduler, img_c, target_c, anchor_handler, loss, plot):
    '''
    img = ({C, H, W}, {C, H, W}, ...)  :tuple
    target = (target1, target2, ...)   :tuple

    target["labels"] = {N, n_box}
    target["boxes"] = {N, n_box, 4}
    target["masks"] = {N, n_box, H, W}
    '''
    device_img = img_c.to(device)
    # {"boxes": boxes, "labels": labels, "masks": masks}

    device_target = []

    for one_target in target_c:
        target = one_target
        target_box = target["boxes"].to(device)
        target_label = target["labels"].to(device)
        target_mask = target["masks"].to(device)

        target["boxes"] = target_box.unsqueeze(0)
        target["labels"] = target_label.unsqueeze(0)
        target["masks"] = target_mask.unsqueeze(0)

        device_target.append(one_target)

    device_target = tuple(device_target)

    optimizer.zero_grad()

    total_loss = loss(device, model, device_img, device_target, anchor_handler, plot=plot)

    if total_loss != 0:
        total_loss.backward()
        optimizer.step()
        # lr_scheduler.step()

    return total_loss


for epoch in range(cfg.total_epoch):
    epoch_time = time.time()

    for iter, (img, target) in enumerate(data_loader_train):
        '''
        img = ({C, H, W}, {C, H, W}, ...)
        target = (target1, target2, ...)
        
        target["labels"] = {N, n_box}
        target["boxes"] = {N, n_box, 4}
        target["masks"] = {N, n_box, H, W}
        '''
        img = torch.stack(img)  # img = {N, C, H, W}

        cur = time.time()
        total_loss = train(cfg.device, optimizer, lr_scheduler, img, target, anchor_handler, loss,
                           True if iter == len(data_loader_train) - 1 else False)

        print("[{}/{}][{}/{}] train loss : {}, TIME : {}s".format(epoch + 1, cfg.total_epoch, iter + 1, len(data_loader_train), total_loss, time.time() - cur))
    print("EPOCH TIME : {}s".format(time.time() - epoch_time))

    if epoch == cfg.total_epoch - 1:
        model.print_weights()
        model.save_weights("/home/user01/data_ssd/LeeJongHyeok/pytorch_project/YOLACT/model/model_epoch_200_lr_1e-4_bat_20_layer_2to5.pth")
