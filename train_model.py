import torch
import data.DataHandler as data_handle
import utils.config as cfg
from model.yolacat import YOLACT as yolact
from structure.LossAll import AllLoss
from utils.AnchorHandler import AnchorHandler as anchor_handle
import time

data_loader_train, data_loader_test = data_handle.get_data(cfg.dataset)

model = yolact(cfg.device)
model.to(cfg.device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

loss = AllLoss()
anchor_handler = anchor_handle(img_h=cfg.image_height, img_w=cfg.image_width)
anchor_handler.cal_anchor()


def train(device, optimizer, lr_scheduler, img_c, target_c, anchor_handler, loss, plot):
    img = img_c.to(device)
    # {"boxes": boxes, "labels": labels, "masks": masks}

    target = target_c
    target_box = target["boxes"].to(device)
    target_label = target["labels"].to(device)
    target_mask = target["masks"].to(device)

    target["boxes"] = target_box
    target["labels"] = target_label
    target["masks"] = target_mask

    optimizer.zero_grad()

    total_loss = loss(device, model, img, target, anchor_handler, plot=plot)

    total_loss.backward()
    optimizer.step()
    lr_scheduler.step()

    return total_loss


# train
model.train()

for epoch in range(cfg.total_epoch):
    for iter, (img, target) in enumerate(data_loader_train):

        cur = time.time()
        total_loss = train(cfg.device, optimizer, lr_scheduler, img, target, anchor_handler, loss,
                           True if epoch == cfg.total_epoch - 1 or iter == len(data_loader_train) - 1 else False)

        print("[{}/{}][{}/{}] train loss : {}, TIME : {}s".format(epoch, cfg.total_epoch, iter, len(data_loader_train), total_loss, time.time() - cur))
