import torch
import data.DataHandler as data_handle
import utils.config as cfg

data_loader_train, data_loader_test = data_handle.get_data(cfg.dataset)

mean = 0.
std = 0.
nb_samples = 0.
for data in data_loader_train:
    batch_samples = data[0][0].size(0)
    data = data[0][0].view(batch_samples, data[0][0].size(1), -1)
    mean += data.mean(2).mean(1)
    std += data.std(0).sum()
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

print("mean", mean)
print("std", std)
