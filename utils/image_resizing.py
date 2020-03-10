from PIL import Image
import os

# path = '../data/PennFudanPed/img'
path = '../data/PennFudanPed/mask'
lists = os.listdir(path)

for i in lists:
    img = os.path.join(path, i)
    print(img)
    image = Image.open(img)
    resize_image = image.resize((512, 512))
    resize_image.save(img)
