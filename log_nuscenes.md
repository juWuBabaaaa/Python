### mmdet3d on nuscenes on create data: 

```FileNotFoundError: file "./data/nuscenes/samples/LIDAR_TOP/n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385112148884.pcd.bin" does not exist```

nuscenes 标注文件有12个json文件，通过逐个查看，sample.json似乎是第一个头绪。需要着重看一下。

### To reduce the sample.json

已经找到报错ID位置，只需要掌握保存json的代码即可。

### After reduce the sample.json, new ID tokens error occured!

(WRONG! 该操作使用的是缩减后的文件)经排查，该token不在sample.json里面。感觉需要更细致的reduce。次可能出现的地方是sample_annotation.json. 


# Record of stastic analysis on *.json files

### calibrated_sensor.json

It has **10200** tokens, 10200 sample_tokens, but 12 unique sample_tokens.

### category.json

nothing need to say, 23 categories.

### Sample_annotation.json

**1166187** annotations from **34140** samples. cover **64386** instances, with 4 degrees visibilities. (**Note: 34140 occurs again.**)

(**Conclusion**: *sample_annotation.json* also need to be reduced when reducing *sample.json*)

### instance.json

**64386** intances cover **23** categories. (first? last?)

### map.json

**4** maps.

### sample_data.json

2631083 items, from **34149** samples, 2631083 ego_pose, 10200 calibrated_sensors, 2630531 timestamps.

**Note**: samples number doesn't match with the number above.

**Curious** about the relationship among *sample*, *sample_data*, *sample_annotation*.

### ego_pose.json
 **2631083** items