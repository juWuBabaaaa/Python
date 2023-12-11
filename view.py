import cv2
import os

dn = '/home/wsx/pointcloud_research/data/nuscenes/samples'
lidar_path = os.path.join(dn, 'LIDAR_TOP')
img_path = os.path.join(dn, "CAM_FRONT")

lidar_files = sorted(os.listdir(lidar_path))
img_files = sorted(os.listdir(img_path))

## 1. view image
# for im in img_files: 
#     mat = cv2.imread(os.path.join(img_path, im))
#     cv2.imshow('img', mat)
#     cv2.waitKey(60)

