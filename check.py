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
with open(anno_p) as w:
    anno = json.load(w)

with open(data_p) as w:
    data = json.load(w)

with open(sample_p) as w:
    sample = json.load(w)

anno_next = []