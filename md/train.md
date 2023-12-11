# Data3dPreprocessing 

"""Points / Image pre-processor for point clouds / vision-only / multi-
    modality 3D detection tasks.

    It provides the data pre-processing as follows

    - Collate and move image and point cloud data to the target device.

    - 1) For image data:

      - Pad images in inputs to the maximum size of current batch with defined
        ``pad_value``. The padding size can be divisible by a defined
        ``pad_size_divisor``.
      - Stack images in inputs to batch_imgs.
      - Convert images in inputs from bgr to rgb if the shape of input is
        (3, H, W).
      - Normalize images in inputs with defined std and mean.
      - Do batch augmentations during training.

    - 2) For point cloud data:

      - If no voxelization, directly return list of point cloud data.
      - If voxelization is applied, voxelize point cloud according to
        ``voxel_type`` and obtain ``voxels``.


# data(dict, two keys)

* data_samples
* inputs

## inputs (dict, one key)

* points list of tensor (exactly the loaded raw pcd files)

# voxelize (function) (pcd data need this process.)

## args: 
* points (List[Tensor]): Point cloud in one data batch.

* data_samples: the annotation data of every samples. Add voxel-wise annotation for segmentation.

## returns:

Dict[str, Tensor]: Voxelization information.

- voxels (Tensor): Features of voxels, shape is MxNxC for hard
    voxelization, NxC for dynamic voxelization.
- coors (Tensor): Coordinates of voxels, shape is Nx(1+NDim),
    where 1 represents the batch index.
- num_points (Tensor, optional): Number of points in each voxel.
- voxel_centers (Tensor, optional): Centers of voxels.

## max_voxels 30000

## voxel size: [0.25, 0.25, 8]

## coor_range: [-50, -50, -5, 50, 50, 3]

**从参数上看到，是圆柱. 但是应该有16w个方格，怎么筛选以满足3w的设定.**

## voxels:

* shape: (max_voxels, max_points, points.size[1])

eg: (30000, 64, 4) --将点云中的点放入到这样一个tensor里.

## coors:

*shape:(max_voxels, 3), dtype: int

eg: (30000, 3)

每个voxel有一个空间坐标.

## num_points_per_voxel

(30000)

# Big Question: ext_loader.load_ext

**由于调试时无法step into这个函数，所以不知道他是怎么实现voxelize的。**

# Runner

## args

* train_dataloader, train_cfg
* test_dataloader, test_cfg
* val_dataloader, val_cfg

## Finaly, I find these two lines


```
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)
```

## data_batch

keys:

* data_samples (list of Det3DDataSample objects)
* inputs
    * key:points (pcd file)


## function: run_iter 
Iterate one min-batch.

### function: self.runner.model.train_step

为什么losses['cls']中只有3个数值？

## The next nut is: 

self._run_forward(data, mode='loss')

## MVX

MVXTwoStageDetector （object）

Multi-modality VoxelNet

### extract_feat

#### extract_pts_feat, args:
* voxel_dict
  * 'num_points': (7020)
  * 'voxel_centers': (7020, 3)
  * 'voxels': (7020, 64, 4)
  * 'coors': (7020, 4)
* points (points in pcd file)
* batch_input_metas
  * 'pcd_scale_factor'
  * 'pcd_trans', 
  * 'lidar_path', 
  * 'pcd_rotation_angle', 
  * 'pcd_horizontal_flip', 
  * 'sample_idx', 
  * 'cam2img', 
  * 'transformation_3d_flow', 
  * 'box_type_3d', 
  * 'lidar2cam', 
  * 'num_pts_feats', 
  * 'ego2global', 
  * 'img_path', 
  * 'pcd_vertical_flip', 
  * 'pcd_rotation', 
  * 'box_mode_3d'

#### class: HardVFE

Voxel feature encoder. (DV-SECOND)

"""Voxel feature encoder used in DV-SECOND.

    It encodes features of voxels and their points. It could also fuse
    image feature into voxel features in a point-wise manner.

    Args:
        in_channels (int, optional): Input channels of VFE. Defaults to 4.
        feat_channels (list(int), optional): Channels of features in VFE.
        with_distance (bool, optional): Whether to use the L2 distance
            of points to the origin point. Defaults to False.
        with_cluster_center (bool, optional): Whether to use the distance
            to cluster center of points inside a voxel. Defaults to False.
        with_voxel_center (bool, optional): Whether to use the distance to
            center of voxel for each points inside a voxel. Defaults to False.
        voxel_size (tuple[float], optional): Size of a single voxel.
            Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): The range of points
            or voxels. Defaults to (0, -40, -3, 70.4, 40, 1).
        norm_cfg (dict, optional): Config dict of normalization layers.
        mode (str, optional): The mode when pooling features of points inside a
            voxel. Available options include 'max' and 'avg'.
            Defaults to 'max'.
        fusion_layer (dict, optional): The config dict of fusion layer
            used in multi-modal detectors. Defaults to None.
        return_point_feats (bool, optional): Whether to return the
            features of each points. Defaults to False.
    """

##### The forwarad function

tensor 中的一些函数：
type_as
keep_dim 作用不大，但是十分有用。

Given a data of point cloud points, it may have the shape of $(m, 4)$, where $m$ is the number of points. 

###### voxelize process, 

**Below operations are handled on one pcd file.**

A tensor box with shape $(M, N, 4)$, where $M$ is the number of voxels predefined. $N$ is the maximum number of points in each voxel. The number $4$ in the third dimension stands for (x, y, z, r). Hence the reflection variable doesn't contribute to the space information. So, next we ignore it in this step. A voxelized data is represented with $X$， with shape $(M, N, 3)$.

Another tensor $Y$, with shape $(M)$ records the number of points in each voxel.

Using this type of tensor, the voxelize process of point cloud points is realized. 

Sum $X$ in dimension 1, a new tensor $X_s$ with shape $(M, 1, 3)$ is obtained. It stores the mass of each voxel. 

We can write $X_s$ in vector form:

$$X_s=(s_1, s_2, \cdots, s_M)$$

where $p_i=(x_i, y_i, z_i)$ is the mass of all points in voxel $i$.

Similarly, write $Y$ as 
$$Y=(y_1, y_2, \cdots, y_M)$$

where $y_i$ represents the number of point in voxel $i$.

Then we get the center of each voxel, 

$$X_c=X_s / Y = (c_1, c_2, \cdots, c_M).$$

到这里为止，还没有什么了不起的东西出现。但是，这些计算的torch数组实现还是很不错的，有技巧。


###### Some parameters need to know about HardVFE

```
    in_channels
    feat_channels
    with_cluster_center
    with_voxel_center
    voxel_size
    point_cloud_range
    norm_cfg
    mode
    fusion_layer (dict, optional)
    return_point_feats (bool, optional)

```

* voxel_size (0.25, 0.25, 8)
* pcd_range [-50, -50, -5, 50, 50, 3]

```
np.indices(dimensions, dtype=<class 'int'>, sparse=False)
    Return an array representing the indices of a grid.
```





































