import numpy as np
import os
import cv2
from tqdm import tqdm
from PIL import Image

# os.chdir(os.path.dirname(__file__))
root = './npyfile_cui/'
segs = os.listdir(f'{root}seg/')
ultras = os.listdir(f'{root}ultra/')

segs.sort()
ultras.sort()
print(segs)
print(ultras)
count = 0
for item in zip(segs, ultras):
    seg, ultra = item
    seg_ndar = np.load(f'{root}seg/{seg}')
    ultra_ndar = np.load(f'{root}ultra/{ultra}')
    for i in tqdm(range(seg_ndar.shape[0])):
        seg_data = seg_ndar[i, :, :]
        ultra_data = ultra_ndar[i, :, :]
        # data[data > 0] = 255
        seg_range = np.max(seg_data) - np.min(seg_data)
        ultra_range = np.max(ultra_data) - np.min(ultra_data)

        seg_data = (seg_data - np.min(seg_data)) / seg_range * 255
        ultra_data = (ultra_data - np.min(ultra_data)) / ultra_range * 255

        # print(seg_data.dtype)#64
        # print(ultra_data.dtype)#16

        seg_data = seg_data.transpose(2, 0, 1)
        ultra_data = ultra_data.transpose(2, 0, 1)

        seg_data = seg_data.astype(np.uint8)
        ultra_data = ultra_data.astype(np.uint8)

        image1 = Image.fromarray(seg_data[0])
        image2 = Image.fromarray(ultra_data[0])

        image1.save(f"/home/pose3d/projs/STCN/UNet_Spine_Proj/UNet_Spine/data/masks_cui/{str(count)}.png")
        image2.save(f"/home/pose3d/projs/STCN/UNet_Spine_Proj/UNet_Spine/data/imgs_cui/{str(count)}.png")
        # cv2.imwrite(f"./data/masks/{str(count)}.png", seg_data)
        # cv2.imwrite(f"./data/imgs/{str(count)}.png", ultra_data, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
        count += 1

print("----------------------------------------------------------\n")
print("total number is: " + str(count))
