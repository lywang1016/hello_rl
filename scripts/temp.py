import math
import yaml
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
max_iter = 2000

threshold_his = []
for i in tqdm(range(max_iter)):
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * i / EPS_DECAY)
    threshold_his.append(eps_threshold)


with open('D:\python\code\hello_rl\scripts\config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
lr_his = []
for i in tqdm(range(max_iter)):
    lr = config['lr_end'] + (config['lr_start'] - config['lr_end']) * \
        math.exp(-1. * i / config['lr_decay'])
    lr_his.append(lr)

iteration = np.linspace(start=1, stop=max_iter, num=max_iter)

plt.figure()
plt.plot(iteration, threshold_his, linewidth=2.0)
plt.xlabel('iteration')
plt.ylabel('threshold')
plt.title('Threshold Change')

plt.figure()
plt.plot(iteration, lr_his, linewidth=2.0)
plt.xlabel('iteration')
plt.ylabel('lr')
plt.title('Learning Rate Change')

plt.show()