import torch
import numpy as np
from PIL import Image
import os
import references.detection.transforms as T
import utils.config as cfg
import references.detection.utils as utils
import matplotlib.pyplot as plt


class PennFudanDataset:
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms

        self.imgs = list(sorted(os.listdir(os.path.join(root, "img"))))
        self.masks = list(sorted(os.listdir((os.path.join(root, "mask")))))

    def __getitem__(self, index):
        # one data.
        img_path = os.path.join(self.root, "img", self.imgs[index])
        mask_path = os.path.join(self.root, "mask", self.masks[index])

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("P")

        mask = np.array(mask)

        obj_ids = np.unique(mask)
        # Notice change the background
        # obj_ids = obj_ids[1:]  # except background.
        obj_ids = obj_ids[:-1]  # except background.

        masks = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)

        boxes = []

        for i in range(num_objs):
            obj_pixels = np.where(masks[i])
            xmin = np.min(obj_pixels[1])
            xmax = np.max(obj_pixels[1])
            ymin = np.min(obj_pixels[0])
            ymax = np.max(obj_pixels[0])

            # NOTICE:  boxes : {ch ,cw, h, w}
            boxes.append([(ymax + ymin) / 2, (xmax + xmin) / 2, ymax - ymin, xmax - xmin])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.float32)

        target = {"boxes": boxes, "labels": labels, "masks": masks}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_data(path):
    def get_transform(train):
        transforms = []
        transforms.append(T.ToTensor())
        # transforms.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        return T.Compose(transforms)

    dataset = PennFudanDataset(path, get_transform(train=True))
    dataset_test = PennFudanDataset(path, get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-cfg.data_divide])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-cfg.data_divide:])

    # define training and validation data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.train_batch_size, shuffle=cfg.shake_data, num_workers=cfg.train_num_works, collate_fn=utils.collate_fn)
    # collate_fn :  for variable size of batch. , collate_fn=utils.collate_fn

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0)

    return data_loader_train, data_loader_test
