import json
import os


path = "/home/wsx/pointcloud_research/data/nuscenes/v1.0-trainval"

# calibrated_sensor.json         category.json                  sample_annotation.json         instance.json                  
# map.json                       sample_data.json               sensor.json                    attribute.json                 
# sample.json                    scene.json                     ego_pose.json                  log.json    

# conts = os.listdir(path)
# for i, j in enumerate(conts):
#     print(f'{j: <30}', end=' ')
#     if (i+1) % 4 == 0:
#         print()

fp = os.path.join(path, "calibrated_sensor.json")
f = open(fp)
ff = json.load(f)
print(ff.keys)
f.close()
