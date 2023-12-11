import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._Arrow3D_verts3d = xs, ys, zs
    
    def do_3d_projection(self, render=None):
        xs3d, ys3d, zs3d = self._Arrow3D_verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


v = np.ones(3)
vector = np.array([1, 0, 0])
tmp = np.c_[np.zeros(3), vector]

q = Quaternion(angle=np.pi/2, axis=np.array([0, 0, 1]))
mat = q.rotation_matrix
vector_ = mat.dot(vector)
tmp1 = np.c_[np.zeros(3), vector_]

q2 = Quaternion(angle=np.pi/4, axis=np.array([1, 1, 1]))
mat2 = q2.rotation_matrix
vector_2 = mat2.dot(vector)
tmp2 = np.c_[np.zeros(3), np.ones(3)]
tmp3 = np.c_[np.zeros(3), vector_2]
ax = plt.figure(figsize=(16, 16)).add_subplot(projection='3d')
arrow_prop_dict = dict(mutation_scale=20, arrowstyle='-|>', color='k', shrinkA=0, shrinkB=0)
arrow_prop_dict1 = dict(mutation_scale=20, arrowstyle='-|>', color='b', shrinkA=0, shrinkB=0)
arrow_prop_dict2 = dict(mutation_scale=20, arrowstyle='-|>', color='r', shrinkA=0, shrinkB=0)
arrow_prop_dict3 = dict(mutation_scale=20, arrowstyle='-|>', color='g', shrinkA=0, shrinkB=0)

a = Arrow3D(tmp[0], tmp[1], tmp[2], **arrow_prop_dict)
b = Arrow3D(tmp1[0], tmp1[1], tmp1[2], **arrow_prop_dict1)
c = Arrow3D(tmp2[0], tmp2[1], tmp2[2], **arrow_prop_dict2)
d = Arrow3D(tmp3[0], tmp3[1], tmp3[2], **arrow_prop_dict3)

ax.add_artist(a)
# ax.add_artist(b)
ax.add_artist(c)
ax.add_artist(d)

ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])
ax.set_xlabel("X", fontsize=26)
ax.set_ylabel("Y", fontsize=26)
ax.set_zlabel("Z", fontsize=26)
ax.scatter(0, 0, 0)

plt.show()
