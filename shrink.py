import json
import os
from tqdm import tqdm
from utils.tools import *
import numpy as np

# 1. Load annotation.json, data.json, sample.json
path = "/home/wsx/pointcloud_research/data/nuscenes/v1.0-trainval"
path2 = "tmp/"

anno_p = os.path.join(path, "sample_annotation.json")
data_p = os.path.join(path, "sample_data.json")
sample_p = os.path.join(path, "sample.json")
scene_p = os.path.join(path, "scene.json")
with open(anno_p) as w:
    anno = json.load(w)

with open(data_p) as w:
    data = json.load(w)

with open(sample_p) as w:
    sample = json.load(w)

with open(scene_p) as w:
    scene = json.load(w)

sample_id = [i["token"] for i in sample]
anno_id = [i["sample_token"] for i in anno]
data_id = [i["sample_token"] for i in data]
scene_id_last = [i["last_sample_token"] for i in scene]
scene_id_first = [i["first_sample_token"] for i in scene]

if "ddc53afb470a4c3ead61f4be71e4b51a" in scene_id_first:
    print("scene True")

# # 2. Find True pieces
c,d = [],[]
for item in tqdm(scene):
    if item["last_sample_token"] not in sample_id:
        c.append(False)
    else:
        c.append(True)
for item in tqdm(scene):
    if item["first_sample_token"] not in sample_id:
        d.append(False)
    else:
        d.append(True)

scene_index_last = find_pieces(c)
scene_index_first = find_pieces(d)
print(scene_index_first)
print(scene_index_last)

scene_shrink = scene[:510]
with open("scene.json", 'w') as s:
    json.dump(scene_shrink, s)
# a, b = [], []
# for item in tqdm(anno):
#     if item["sample_token"] not in sample_id:
#         a.append(False)
#     else:
#         a.append(True)

# for item in tqdm(data):
#     if item["sample_token"] not in sample_id:
#         b.append(False)
#     else:
#         b.append(True)

# a = np.array(a)
# b = np.array(b)

# anno_index = find_pieces(a)
# data_index = find_pieces(b)

# print(anno_index)
# print(data_index)

# result: Anno, [[0, 736060], [1165955, 1166187]] data, [[0, 1570195]]

# # 3. Save the shrinked files
# anno_new = anno[:736060] + anno[1165955: 1166187]
# data_new = data[:1570195]

# # print(len(anno_new))
# # print(len(data_new))

# # if len(anno_new) == 736060 + (1166187-1165955):
# #     print(True)

# with open("sample_annotation.json", 'w') as w:
#     json.dump(anno_new, w)

# with open("sample_data.json", 'w') as w:
#     json.dump(data_new, w)

# 4. check ddc53afb470a4c3ead61f4be71e4b51a


# 5. sample.json -> next

print(sample[-1]["next"])

# sample[-1]["next"] = ''
# with open("sample.json", 'w') as f:
#     json.dump(sample, f)

# 6. scene.json
