# 函数调用关系
## create_data.py

* nuscenes_data_prep (basic infos, 2d annotations, gt database)
    * converter.create_nuscenes_infos
        * splits.val
        * get_available_scenes(nusc)
        * _fill_trainval_infos
            * lidarpath, boxes = nusc.get_sample_data(lidar_token) # 点云的信息基本齐了
            * obtain_sensor2top()
            
* update_pkl_infos
  * update_nuscenes_infos
* create_groundtruth_database
  * dataset = DATASETS.build(dataset_cfg)
  * mmdet3d.structures.ops： box_np_ops- points_in_rbbox

### lidarpath, boxes

lidarpath: LIDAR_TOP一帧点云

**box**
* label
* score
* xyz
* wlh
* rot axis 
* ang(degrees)
* ang(rad)
* vel (vx, vy, vz)
* name
* token

### rotation & translation

外参 RT矩阵

```l2e_r = info['lidar2ego_rotation']
l2e_t = info['lidar2ego_translation'] # 3
e2g_r = info['ego2global_rotation']   # 4
e2g_t = info['ego2global_translation']
l2e_r_mat = Quaternion(l2e_r).rotation_matrix  # 3*3
e2g_r_mat = Quaternion(e2g_r).rotation_matrix
```

### obtain_sensor2top function

函数得到sensor 坐标系到点云坐标系的旋转平移矩阵.

Obtain the info with RT matric from general sensor to Top LiDAR.

lidar -> ego -> global


### ego vehicle
**Vehicle coordinate system**

The term ego refers to the vehicle that contains the sensors that perceive the environment around the vehicle.
```
# obtain the RT from sensor to Top LiDAR
# sweep->ego->global->ego'->lidar

```
注意这里两个ego不同
numpy 中@表示 矩阵乘法

### annotation

```
2 0 movable_object.trafficcone
3 0 movable_object.trafficcone
171 7 vehicle.truck
150 2 vehicle.car
7 0 human.pedestrian.construction_worker
151 3 vehicle.car
9 3 vehicle.truck
10 0 human.pedestrian.construction_worker
4 0 movable_object.trafficcone
42 6 vehicle.truck
```

* rots: yaw, pitch, roll 只取了yaw
* velocity: 只取了前两个值
* valid_flag: (anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
* convert velo from global to lidar
  global -> ego -> lidar 只乘了旋转矩阵，最后取前两位。
* we need to convert box size to the format of our lidar coordinate system which is x_size, y_size, z_size (corresponding to **l, w, h**)（注意顺序）

### info 
keys
* 'lidar_path', 
* 'num_features', 
* 'token', 
* 'sweeps', 
* 'cams', 
* 'lidar2ego_translation', 
* 'lidar2ego_rotation', 
* 'ego2global_translation', 
* 'ego2global_rotation', 
* 'timestamp', 
* 'gt_boxes', 
* 'gt_names', 
* 'gt_velocity',
* 'num_lidar_pts', 
* 'num_radar_pts',
* 'valid_flag'

#### Remark

* One of the amazing things is 六个摄像头均计算了摄像头到点云的坐标变换RT参数。到时候怎么用还需落实。
* 

#### gt_boxes
7位：x,y,z,l,w,h,rot
#### gt_names
#### gt_velocity
transform from **global** to **lidar**
(疑问：global速度是怎么获得的？)
#### num_lidar_pts
#### num_radar_pts
radar和lidar的相互关系？
对radar很陌生。
#### valid_flag

### dataset (build dataset)
* 10个类别10种颜色。（palette：调色板）
* num_ins_per_cat (每个种类对象个数。这个字典很关键)
* data address (长度为80（80个sample）的数组)，其中的数值还没看懂什么意思。

### dataset write
* data_info
* example = dataset.pipeline(data_info)
  * pipeline
    * LoadPointsFromFile
    * LoadPointsFromMultiSweeps
    * LoadAnnotations3D
  * example.keys()
    *  sample_idx
    *  token
    *  timestamp
    *  ego2global
    *  images: 图像的一些信息，并没有读取图像
    *  lidar_points
    *  instances
    *  cam_instances
    *  num_pts_feats
    *  lidar_path
    *  ann_info
    *  points: 点云数据，已读取
    *  gt_bboxes_3d: 9位数
    *  gt_labels_3d

### function: points_in_rbbox

Check points in rotated bbox and return indices

This function is for **counterclockwise boxes.**

* rbbox_corners = center_to_corner_box3d(**rbbox[:, :3], rbbox[:, 3:6], rbbox[:, 6]**, origin=origin, axis=z_axis)
* surfaces = corner_to_surfaces_3d(rbbox_corners)
* indices = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)

#### 计算顶点: center_to_corner_box3d (mmdet3d/structures/ops/box_np_ops.py)
   Convert kitti locations, dimensions and angles to corners.
    Returns:
        np.ndarray: Corners with the shape of **(N, 8, 3)**.

#### 计算表面surface: corner_to_surfaces_3dconvert 3d box 

corners from corner function above to surfaces that
    normal vectors all direct to internal.

* input : 3D box corners with shape of (N, 8, 3)
* returns: np.ndarray: Surfaces with the shape of (N, 6, 4, 3). (**6个面，每个面4个点**)

#### 判断每个点是否在凸多面体内：points_in_convex_polygon_3d_jit


Check points is in 3d convex polygons

* input: 
  * points (np.ndarray): Input points with shape of (num_points, 3).
  * polygon_surfaces (np.ndarray): Polygon surfaces with shape of (num_polygon, max_num_surfaces max_num_points_of_surface, 3). All surfaces' normal vector must direct to internal. Max_num_points_of_surface must at least 3.
  * num_surfaces (np.ndarray, optional): Number of surfaces a polygon contains shape of (num_polygon). Defaults to None.

* returns: 
  * np.ndarray: Result matrix with the shape of **[num_points, num_polygon(box numbers)]**.

#### 平移每个物体到中心为原点
gt_points[:, :3] -= gt_boxes_3d[i, :3]

# preprocess data.
* nuscenes_database 一直没搞懂这个什么用。
* Sweeps: The intermediate lidar frames without annotations
* Velocities of 3D bounding boxes (no vertical measurements due to inaccuracy), a list has shape (2,).

# Training pipeline
## Lidar-based
* Compared to general cases, nuScenes has a specific 'LoadPointsFromMultiSweeps' pipeline to load point clouds from consecutive frames. This is a common practice used in this setting. Please refer to the nuScenes original paper for more details. The default use_dim in 'LoadPointsFromMultiSweeps' is [0, 1, 2, 4], where the first 3 dimensions refer to point coordinates and the last refers to timestamp differences. Intensity is not used by default due to its yielded noise when concatenating the points from different frames.
## Vision-based