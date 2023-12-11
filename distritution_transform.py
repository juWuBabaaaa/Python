import numpy as np
import ot
import matplotlib.pyplot as plt
import os
import pandas as pd

path = "/home/wsx/pointcloud_research/data/nuscenes/nuscenes_gt_database/"

df = pd.read_csv("stat_gtdb.csv")
cls = list(pd.unique(df["class"]))
grouped = df.groupby("class").apply(pd.DataFrame.describe)

f = lambda x: x.sort_values(by="num", ascending=False)
grouped1 = df.groupby("class").apply(f)

car = grouped1.xs("car")
x = car["num"].to_numpy()
tmp = np.zeros(len(x))
for i in x:
    tmp[i] += 1

r = dict()
for i, num in enumerate(tmp):
    if num != 0:
        r[i] = num

tmp1 = []                           
for i, num in r.items():
    tmp1.append([i, num])
    print(i, num)
tmp1 = np.array(tmp1)        # the desired array.

p1 = tmp1 / sum(tmp1)

p2 = np.ones_like(p1) * (1. / len(p1))




