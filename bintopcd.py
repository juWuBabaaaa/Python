import numpy as np
import matplotlib.pyplot as plt
import os
import open3d as o3d
from mmdet3d.visualization import Det3DLocalVisualizer
import cv2

# 13398_car_24.bin
# 13407_car_17.bin
#14006_car_2.bin
# 10177_car_14.bin
# 24045_car_14.bin
# '/home/wsx/pointcloud_research/data/nuscenes/nuscenes_gt_database/24045_car_14.bin'
# path = "/home/wsx/pointcloud_research/data/nuscenes/samples/LIDAR_TOP/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin"

dp = "/home/wsx/pointcloud_research/data/nuscenes/samples"
lp = os.path.join(dp, "LIDAR_TOP")
imgp = os.path.join(dp, "CAM_FRONT")

ls = os.listdir(lp)
imgp = os.listdir(imgp)

print(ls[0])
print(imgp[0])
print(len(ls))
print(len(imgp))
f = lambda x: x.split(".")[0]

def fun(x):
    tmp = x.split("__")
    return "_".join(tmp[::2])

ln = list(map(f, ls))
imgn = list(map(f, imgp))

print(ln[0])
print(imgn[0])

ln = list(map(fun, ln))
imgn = list(map(fun, imgn))

ln = sorted(ln)
imgn = sorted(imgn)

print(ln[0])
print(imgn[0])

print(ln == imgn)

# for i in range(80):
#     print(ln[i], "\t", imgn[i])

# view pcd file
with open(path, 'rb') as f:
    a = np.fromfile(f, dtype=np.float32)
b = a.reshape(-1, 5)

p = b[:, :4]
visualizer = Det3DLocalVisualizer()
visualizer.set_points(p)
visualizer.show()

mat = cv2.imread("/home/wsx/pointcloud_research/data/nuscenes/samples/CAM_FRONT/n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915243012465.jpg")

cv2.imshow("a", mat)
cv2.waitKey(0)
