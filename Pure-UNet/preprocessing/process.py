import numpy as np
import os
import cv2
from tqdm import tqdm
from PIL import Image

# os.chdir(os.path.dirname(__file__))
root = './npyfile/'
segs = os.listdir(f'{root}seg_added/')
ultras = os.listdir(f'{root}ultra_added/')

segs.sort()
ultras.sort()
print(segs)
print(ultras)
count = 0
for item in zip(segs, ultras):
    seg, ultra = item
    seg_ndar = np.load(f'{root}seg_added/{seg}')
    ultra_ndar = np.load(f'{root}ultra_added/{ultra}')
    for i in tqdm(range(seg_ndar.shape[0])):
        seg_data = seg_ndar[i, :, :]
        ultra_data = ultra_ndar[i, :, :]
        # data[data > 0] = 255
        seg_range = np.max(seg_data) - np.min(seg_data)
        ultra_range = np.max(ultra_data) - np.min(ultra_data)

        seg_data = (seg_data - np.min(seg_data)) / seg_range * 255
        ultra_data = (ultra_data - np.min(ultra_data)) / ultra_range * 255

        cv2.imwrite(f"/home/pose3d/projs/STCN/UNet_Spine_Proj/UNet_Spine/data/masks/{str(count)}.png", seg_data)
        cv2.imwrite(f"/home/pose3d/projs/STCN/UNet_Spine_Proj/UNet_Spine/data/imgs/{str(count)}.png", ultra_data)
        count += 1

print("----------------------------------------------------------\n")
print("total number is: " + str(count))