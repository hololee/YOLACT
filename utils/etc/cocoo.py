from PIL import Image
import os
import shutil
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

path = '/data/StrawInst'
tar1 = '/home/user01/data_ssd/LeeJongHyeok/pytorch_project/YOLACT/data/StrawInst/img/'
tar2 = '/home/user01/data_ssd/LeeJongHyeok/pytorch_project/YOLACT/data/StrawInst/mask/'
lists1 = os.listdir(tar1)
lists2 = os.listdir(tar2)

for indxm, i in enumerate(lists2):
    im = Image.open(os.path.join(tar2 + i))

    print(im.size)
    im = im.convert('RGB').convert('P', palette=Image.ADAPTIVE)

    im.save(os.path.join(tar2 + i))
    print(indxm)


