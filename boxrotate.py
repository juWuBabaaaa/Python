import pickle
import os
import numpy as np
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from tqdm import tqdm

dp = "/home/wsx/pointcloud_research/data/nuscenes/tmp"
with open(os.path.join(dp, "nuscenes_infos_train.pkl"), 'rb') as f:
    file = pickle.load(f)

metainfo = file["metainfo"]
data_list = file["data_list"]
box = data_list[0]["instances"][0]["bbox_3d"]

def convertBox(box):
    x, y, z, l, w, h, yaw = box
    corners = np.array(
        [
            [x-l/2, y-w/2, z-h/2],
            [x-l/2, y-w/2, z+h/2],
            [x-l/2, y+w/2, z-h/2],
            [x-l/2, y+w/2, z+h/2],
            [x+l/2, y-w/2, z-h/2], 
            [x+l/2, y-w/2, z+h/2],
            [x+l/2, y+w/2, z-h/2],
            [x+l/2, y+w/2, z+h/2]
        ]
    )
    corners[:, 0] -= x
    corners[:, 1] -= y
    corners[:, 2] -= z

    q = Quaternion(angle=yaw, axis=np.array([0, 0, 1]))
    # print(q.rotation_matrix)
    corners1 = (q.rotation_matrix.dot(corners.T)).T
    corners1[:, 0] += x
    corners1[:, 1] += y
    corners1[:, 2] += z
    xmin, ymin, zmin = corners1.min(axis=0)
    xmax, ymax, zmax = corners1.max(axis=0)
    return xmin, xmax, ymin, ymax, zmin, zmax

dp = "/home/wsx/pointcloud_research/data/nuscenes/anno_cplus_minmax"
for item in tqdm(data_list):
    fp = item["lidar_points"]["lidar_path"] + ".txt"
    with open(os.path.join(dp, fp), 'w') as f:
        for i in item["instances"]:
            f.write(str(i["bbox_label"])+" ")
            box = i["bbox_3d"]
            re = convertBox(box)
            for k in re:
                f.write(str(round(k, 2)) + " ")
            f.write("\n")
    # print(corners.min(axis=0))
    # print(corners.max(axis=0))

    # fig = plt.figure(figsize=(20, 20))
    # ax = fig.add_subplot(projection='3d')
    # ax.set_xlim([-1, 1])
    # ax.set_ylim([-1, 1])
    # ax.set_zlim([-1, 1])
    # for i in range(8):
    #     ax.plot(corners[i, 0], corners[i, 1], corners[i, 2], '.', markersize=20, label=f"{i}")
    # ax.plot(corners[[0, 1], 0], corners[[0, 1], 1], corners[[0, 1], 2], color='b', linewidth=5)
    # ax.plot(corners[[1, 3], 0], corners[[1, 3], 1], corners[[1, 3], 2], color='b', linewidth=5)
    # ax.plot(corners[[3, 2], 0], corners[[3, 2], 1], corners[[3, 2], 2], color='b', linewidth=5)
    # ax.plot(corners[[2, 0], 0], corners[[2, 0], 1], corners[[2, 0], 2], color='b', linewidth=5)
    # ax.plot(corners[[0, 4], 0], corners[[0, 4], 1], corners[[0, 4], 2], color='b', linewidth=5)
    # ax.plot(corners[[4, 6], 0], corners[[4, 6], 1], corners[[4, 6], 2], color='b', linewidth=5)
    # ax.plot(corners[[6, 2], 0], corners[[6, 2], 1], corners[[6, 2], 2], color='b', linewidth=5)
    # ax.plot(corners[[1, 5], 0], corners[[1, 5], 1], corners[[1, 5], 2], color='b', linewidth=5)
    # ax.plot(corners[[5, 7], 0], corners[[5, 7], 1], corners[[5, 7], 2], color='b', linewidth=5)
    # ax.plot(corners[[7, 3], 0], corners[[7, 3], 1], corners[[7, 3], 2], color='b', linewidth=5)
    # ax.plot(corners[[4, 5], 0], corners[[4, 5], 1], corners[[4, 5], 2], color='b', linewidth=5)
    # ax.plot(corners[[6, 7], 0], corners[[6, 7], 1], corners[[6, 7], 2], color='b', linewidth=5)

    # ax.plot(corners1[[0, 1], 0], corners1[[0, 1], 1], corners1[[0, 1], 2], color='r', linewidth=5)
    # ax.plot(corners1[[1, 3], 0], corners1[[1, 3], 1], corners1[[1, 3], 2], color='r', linewidth=5)
    # ax.plot(corners1[[3, 2], 0], corners1[[3, 2], 1], corners1[[3, 2], 2], color='r', linewidth=5)
    # ax.plot(corners1[[2, 0], 0], corners1[[2, 0], 1], corners1[[2, 0], 2], color='r', linewidth=5)
    # ax.plot(corners1[[0, 4], 0], corners1[[0, 4], 1], corners1[[0, 4], 2], color='r', linewidth=5)
    # ax.plot(corners1[[4, 6], 0], corners1[[4, 6], 1], corners1[[4, 6], 2], color='r', linewidth=5)
    # ax.plot(corners1[[6, 2], 0], corners1[[6, 2], 1], corners1[[6, 2], 2], color='r', linewidth=5)
    # ax.plot(corners1[[1, 5], 0], corners1[[1, 5], 1], corners1[[1, 5], 2], color='r', linewidth=5)
    # ax.plot(corners1[[5, 7], 0], corners1[[5, 7], 1], corners1[[5, 7], 2], color='r', linewidth=5)
    # ax.plot(corners1[[7, 3], 0], corners1[[7, 3], 1], corners1[[7, 3], 2], color='r', linewidth=5)
    # ax.plot(corners1[[4, 5], 0], corners1[[4, 5], 1], corners1[[4, 5], 2], color='r', linewidth=5)
    # ax.plot(corners1[[6, 7], 0], corners1[[6, 7], 1], corners1[[6, 7], 2], color='r', linewidth=5)
    # plt.legend(loc="best")
    # plt.show()




