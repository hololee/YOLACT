a = 3  # number of anchors.
num_classes = 1  # number of classes.
num_prototype = 6  # number of prototypes.
default_detect_iou = 0.5  # define isObject threshold.

pre_trained = False
total_epoch = 200  # epochs.
train_batch_size = 1  # batch_size.
train_num_works = 4  # for debug set 0

loss_cls_alpha = 1

shake_data = False

# p3 p4 p5 p6 p7
using_start_layer_no = 2
using_stop_layer_no = 5

# A1 case 0:3
# in person case 2:5

# dataset = './data/A1'
# dataset = './data/PennFudanPed'
dataset = './data/StrawInst'

data_divide = 50