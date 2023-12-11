import torch
from torch import nn
from mmengine import Registry
import matplotlib.pyplot as plt

ACTIVATION = Registry('activation', scope='mmengine', locations=['mmengine.models.activations'])


act_cfg = dict(type='Sigmoid')
activation = ACTIVATION.build(act_cfg)
input = torch.linspace(-3, 3, 100)
output = activation(input)
plt.plot(input, output)
plt.show()
