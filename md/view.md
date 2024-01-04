# visualize
MMDetection3D提供了一个Det3DLocalVisualizer来可视化和存储模型在训练和测试期间的状态，以及结果，具有以下功能。

* 支持多模态数据和多任务的基本绘图接口。


* 支持local, TensorBoard等多个后端，将loss, lr或性能评估指标等训练状态写入指定的单个或多个后端。


* 支持多模态数据的地面真值可视化和三维检测结果的跨模态可视化。


Det3DLocalVisualizer继承自DetLocalVisualizer，提供了在2D图像上绘制常见对象的接口，例如绘制检测框、点、文本、线、圆、多边形和二进制蒙版。关于2D绘图的更多细节可以参考MMDetection中的可视化文档。下面介绍三维绘图界面:

* Drawing point cloud on the image
* Drawing 3D Boxes on Point Cloud
* Drawing Projected 3D Boxes on Image
* Drawing BEV Boxes
* Drawing 3D Semantic Mask
* Prediction results
* Dataset online visualization


Det3DLocalVisualizer(DetLocalVisualizer)

- draw_bboxes_3d: **draw 3D bounding boxes on point clouds**
- draw_proj_bboxes_3d: **draw projected 3D bounding boxes on image**
- draw_seg_mask: **draw segmentation mask via per-point colorization**

