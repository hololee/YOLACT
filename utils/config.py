import torch

image_height = 512
image_width = 512

a = 3  # number of anchors.
num_classes = 1  # number of classes.
num_prototype = 32  # number of prototypes. k
default_detect_iou = 0.5  # define isObject threshold.

pre_trained = False
total_epoch = 1000  # epochs.
learning_rate = 0.0001

# Notice : now not using lr schedule.
step_size = 100
weight_decay = 0.00005

train_batch_size = 20  # batch_size. # 20
train_num_works = 0  # for debug set 0

loss_cls_alpha = 1

shake_data = False

# p3 p4 p5 p6 p7
using_start_layer_no = 1
using_stop_layer_no = 3

# A1 case 0:3
# in person case 2:5

# dataset = '/home/user01/data_ssd/LeeJongHyeok/pytorch_project/YOLACT/data/A1'
# dataset = '/home/user01/data_ssd/LeeJongHyeok/pytorch_project/YOLACT/data/PennFudanPed'
dataset = '/home/user01/data_ssd/LeeJongHyeok/pytorch_project/YOLACT/data/StrawInst'

data_divide = 50

# gpu number
device = torch.device(1 if torch.cuda.is_available() else torch.device('cpu'))

# predict.
pr_epoch = 500  # 500
pred_th = .6

# fast nms threshold
fnms_th = 0.3
top_k = 10
