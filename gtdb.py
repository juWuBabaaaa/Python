import os
import pickle
import numpy as np

fp = "/home/wsx/pointcloud_research/data/nuscenes/"
dname = "nuscenes_gt_database"
data = os.listdir(os.path.join(fp, dname))

fname = "nuscenes_dbinfos_train.pkl"
with open(os.path.join(fp, fname), 'rb') as f:
    train_info = pickle.load(f)

tmp = []
for m, n in train_info.items():
    tmp.append(np.array(list(n[0].keys())))

print(np.unique(tmp))
for i in np.unique(tmp):
    print(i)

