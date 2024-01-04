import json
import os
from tqdm import tqdm
from utils.tools import *
import numpy as np

# 1. Load annotation.json, data.json, sample.json
path = "/home/wsx/pointcloud_research/data/nuscenes/tmp_json/v1.0-trainval"

anno_p = os.path.join(path, "sample_annotation.json")
data_p = os.path.join(path, "sample_data.json")
sample_p = os.path.join(path, "sample.json")
scene_p = os.path.join(path, "scene.json")
category_p = os.path.join(path, "category.json")
instance_p = os.path.join(path, "instance.json")
with open(anno_p) as w:
    anno = json.load(w)

with open(data_p) as w:
    data = json.load(w)

with open(sample_p) as w:
    sample = json.load(w)

with open(scene_p) as w:
    scene = json.load(w)
# 2. select scene[0]
# remark: scene0: 0~39; scene1: 40~79

# 3. select sample
# remark: anno[:1753]
# remark: data[:6135]
# scene keys:   'token', 'log_token', 'nbr_samples', 'first_sample_token', 'last_sample_token', 
#               'name', 'description'
# anno keys:    'token', 'sample_token', 'instance_token', 'visibility_token', 'attribute_tokens',
#               'translation', 'size', 'rotation', 'prev', 'next', 'num_lidar_pts', 'num_radar_pts'
# data keys:    'token', 'sample_token', 'ego_pose_token', 'calibrated_sensor_token', 'timestamp', 
#               'fileformat', 'is_key_frame', 'height', 'width', 'filename', 'prev', 'next'
scene_new = scene[:68]
# # to detect selected scenes.
# print(scene[0].keys())
# for j, i in enumerate(scene_new):
#     print(j, " ", i["nbr_samples"], " ", i["description"])
lt = scene_new[-1]["last_sample_token"]
for i, j in enumerate(sample):
    if j["token"] == lt:
        indice_sample = i+1
        print("sample", indice_sample)


sample_new = sample[:indice_sample]  # next

sample_new_id = [i["token"] for i in sample_new]
tmp = []
for i, j in enumerate(anno):
    if j["sample_token"] in sample_new_id:
        tmp.append(i)
indice_anno = len(tmp)
print("anno ", indice_anno)
tmp1 = []
for i, j in enumerate(data):
    if j["sample_token"] in sample_new_id:
        tmp1.append(i)
indice_data = len(tmp1)
print("data ", indice_data)
anno_new = anno[:indice_anno]
data_new = data[:indice_data]

sample_new[-1]["next"] = ""

# 4. write
with open("sample_annotation.json", 'w') as w:
    json.dump(anno_new, w)

with open("sample_data.json", 'w') as w:
    json.dump(data_new, w)

with open("sample.json", 'w') as f:
    json.dump(sample_new, f)

# # 6. scene.json
with open("scene.json", 'w') as f:
    json.dump(scene_new, f)