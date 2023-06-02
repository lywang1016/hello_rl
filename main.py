import gym
import os
import math
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
from collections import namedtuple, deque
from tqdm import tqdm
from heapq import heapify, heappop, heappush
import torch
import torch.nn as nn
from network import DQN
from reward import reward_function_1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

env = gym.make(config['game'])
observation, info = env.reset()

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
memory = ReplayMemory(10000)
    
BATCH_SIZE = config['batch_size']
GAMMA = config['gamma']
EPS_START = config['eps_start']
EPS_END = config['eps_end']
EPS_DECAY = config['eps_decay']
SAVE_EVERY = config['save_every']
steps_done = 0

n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations+1).to(device)
target_net = DQN(n_observations+1).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = torch.optim.AdamW(policy_net.parameters(), lr=float(config['lr_start']), amsgrad=True)











# for _ in range(1000):
# #    action = policy(observation)  # User-defined policy function 
#     action = env.action_space.sample()
#     observation, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         observation, info = env.reset()
#         ic(observation)
#         ic(reward)
#         ic(terminated)
#         ic(truncated)
#         ic(info)
# env.close()
