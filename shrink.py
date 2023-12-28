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
scene_new = scene[:2]
sample_new = sample[:80]  # next

sample_new_id = [i["token"] for i in sample_new]
anno_new = anno[:1753]
data_new = data[:6135]

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