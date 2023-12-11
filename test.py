import numpy as np
import torch
import matplotlib.pyplot as plt

# # 1. grid picture
# x = np.arange(0, 100, 0.25)
# y = np.arange(0, 100, 0.25)
# xx, yy = np.meshgrid(x,y)
# zz = np.zeros_like(xx)

# ax = plt.figure(figsize=(10, 10)).add_subplot(projection='3d')
# ax.plot_wireframe(xx, yy, zz, rstride=10, cstride=10)
# ax.set_xlim((0, 100))
# ax.set_ylim((0, 100))
# ax.set_zlim((0, 8))

# plt.show()

# # 2. viewlizer of mmdet3d

# import numpy as np
# import torch
# from mmengine.structures import InstanceData
# from mmdet3d.structures import DepthInstance3DBoxes, Det3DDataSample
# from mmdet3d.visualization import Det3DLocalVisualizer

# det3d_local_visualizer = Det3DLocalVisualizer()
# image = np.random.randint(0, 256, size=(10, 12, 3)).astype('uint8')
# points = np.random.rand(1000, 3)
# gt_instances_3d = InstanceData()
# gt_instances_3d.bboxes_3d = DepthInstance3DBoxes(torch.rand((5, 7)))
# gt_instances_3d.labels_3d = torch.randint(0, 2, (5,))
# gt_det3d_data_sample = Det3DDataSample()
# gt_det3d_data_sample.gt_instances_3d = gt_instances_3d
# data_input = dict(img=image, points=points)
# det3d_local_visualizer.add_datasample('3D Scene', data_input, gt_det3d_data_sample)

# from mmdet3d.structures import PointData
# det3d_local_visualizer = Det3DLocalVisualizer()
# points = np.random.rand(1000, 3)
# gt_pts_seg = PointData()
# gt_pts_seg.pts_semantic_mask = torch.randint(0, 10, (1000, ))
# gt_det3d_data_sample = Det3DDataSample()
# gt_det3d_data_sample.gt_pts_seg = gt_pts_seg
# data_input = dict(points=points)
# det3d_local_visualizer.add_datasample('3D Scene', data_input, gt_det3d_data_sample, vis_task='lidar_seg')

# 3. print dir contents
import os
dp = "/usr/include/pcl-1.10/pcl/"
f = os.listdir(dp)
tmp = []
for i in f:
    if len(i.split(".")) == 1:
        tmp.append(i)



with open("tmp.txt", 'w') as f:
    for i in sorted(tmp):
        p = os.path.join(dp, i)
        f.write('"' + p + '",' + '\n')
