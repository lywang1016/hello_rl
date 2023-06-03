import gym
import os
import math
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from icecream import ic
from collections import namedtuple, deque
from tqdm import tqdm
from heapq import heapify, heappop, heappush
import torch
import torch.nn as nn
from network import DQN
from reward import reward_function_1

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

env = gym.make(config['game'])
observation, info = env.reset()

# Transition = namedtuple('Transition',
#                         ('state', 'action', 'reward', 'next_state'))
# class ReplayMemory(object):
#     def __init__(self, capacity):
#         self.memory = deque([], maxlen=capacity)

#     def push(self, *args):
#         """Save a transition"""
#         self.memory.append(Transition(*args))

#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)

#     def __len__(self):
#         return len(self.memory)
# memory = ReplayMemory(10000)
    
# BATCH_SIZE = config['batch_size']
# GAMMA = config['gamma']
# EPS_START = config['eps_start']
# EPS_END = config['eps_end']
# EPS_DECAY = config['eps_decay']
# SAVE_EVERY = config['save_every']
# steps_done = 0

# n_actions = env.action_space.n
# state, info = env.reset()
# n_observations = len(state)

# policy_net = DQN(n_observations+1).to(device)
# target_net = DQN(n_observations+1).to(device)
# target_net.load_state_dict(policy_net.state_dict())

# optimizer = torch.optim.AdamW(policy_net.parameters(), lr=float(config['lr_start']), amsgrad=True)

# def select_action(state):
#     global steps_done
#     global n_actions
#     global policy_net

#     sample = random.random()
#     eps_threshold = EPS_END + (EPS_START - EPS_END) * \
#         math.exp(-1. * steps_done / EPS_DECAY)
#     steps_done += 1
#     if sample > eps_threshold:
#         state_ = torch.from_numpy(state).to(device)
#         state_ = state_.view(1, state_.shape[0])    #shape:[1, 4]
#         queue = []
#         heapify(queue)
#         for i in range(n_actions):
#             a = torch.tensor(i).to(device)
#             a_ = a.float().view(1, 1)
#             value = policy_net(state_, a_)
#             value = value.cpu().detach().numpy()[0][0]
#             heappush(queue, (-value, i))
#         value, a = heappop(queue)
#         return a
#     else:
#         return np.random.randint(n_actions)

# episode_durations = []
# def plot_durations(show_result=False):
#     plt.figure(1)
#     durations_t = torch.tensor(episode_durations, dtype=torch.float)
#     if show_result:
#         plt.title('Result')
#     else:
#         plt.clf()
#         plt.title('Training...')
#     plt.xlabel('Episode')
#     plt.ylabel('Duration')
#     plt.plot(durations_t.numpy())
#     # Take 100 episode averages and plot them too
#     if len(durations_t) >= 100:
#         means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
#         means = torch.cat((torch.zeros(99), means))
#         plt.plot(means.numpy())

#     plt.pause(0.001)  # pause a bit so that plots are updated
#     if is_ipython:
#         if not show_result:
#             display.display(plt.gcf())
#             display.clear_output(wait=True)
#         else:
#             display.display(plt.gcf())

# def optimize_model():
#     global n_actions
#     global policy_net
#     global target_net

#     if len(memory) < BATCH_SIZE:
#         return 1

#     transitions = memory.sample(BATCH_SIZE)
#     batch = Transition(*zip(*transitions))

#     state = torch.from_numpy(batch.state[0]).to(device)
#     action = torch.tensor(batch.action[0]).to(device)
#     reward = torch.tensor(batch.reward[0]).to(device)
#     state_batch = state.view(1, state.shape[0])
#     action_batch = action.view(1, 1)
#     reward_batch = reward.view(1, 1)
#     for i in range(1, BATCH_SIZE):
#         state = torch.from_numpy(batch.state[i]).to(device)
#         action = torch.tensor(batch.action[i]).to(device)
#         reward = torch.tensor(batch.reward[i]).to(device)
#         state_batch = torch.cat((state_batch, state.view(1, state.shape[0])), dim=0)
#         action_batch = torch.cat((action_batch, action.view(1, 1)), dim=0)
#         reward_batch = torch.cat((reward_batch, reward.view(1, 1)), dim=0)

#     next_values = torch.zeros((BATCH_SIZE,1), device=device)
#     for i in range(BATCH_SIZE):
#         next_state = torch.from_numpy(batch.next_state[i]).to(device)
#         next_state = next_state.view(1, next_state.shape[0])
#         val_list = []
#         for j in range(n_actions):
#             a = torch.tensor(j).to(device)
#             a = a.float().view(1, 1)
#             value = target_net(next_state, a)
#             value = value.cpu().detach().numpy()[0][0]
#             val_list.append(value)
#         max_val = max(val_list)
#         next_values[i][0] = torch.tensor(max_val).to(device)
#     next_values = (next_values * GAMMA) + reward_batch

#     cur_values = policy_net(state_batch, action_batch)

#     criterion = nn.SmoothL1Loss()
#     loss = criterion(cur_values, next_values)

#     optimizer.zero_grad()
#     loss.backward()
#     torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
#     optimizer.step()
    
#     return float(loss.cpu().detach())

# # Training
# num_episodes = 600
# duration_history = []
# loss_history = []
# mx_history = []
# good_cnt = 0
# for i_episode in tqdm(range(num_episodes)):
#     if good_cnt>20:
#         num_episodes = i_episode
#         break
#     obs, info = env.reset()
#     done = False 
#     life_cnt = 0
#     total_loss = 0  
#     max_x = 0
#     while not done:
#         action = select_action(obs)
#         new_obs, reward, done, truncated, info = env.step(action)
#         if abs(new_obs[0]) > max_x:
#             max_x = abs(new_obs[0])
#         reward = reward_function_1(new_obs[0], new_obs[2])
#         env.render()
#         life_cnt += 1
#         memory.push(obs, action, reward, new_obs)
#         loss = optimize_model()
#         total_loss += loss
#         obs = new_obs
#         if life_cnt > 1000:
#             good_cnt += 1
#             break
#     if life_cnt < 1000:
#         good_cnt = 0
#     print(' Last episode life time is: ' + str(life_cnt))
#     duration_history.append(life_cnt)
#     print('Last episode loss is: ' + str(total_loss/life_cnt))
#     loss_history.append(total_loss/life_cnt)
#     print('Last episode max x is: ' + str(max_x))
#     mx_history.append(max_x)
#     if i_episode % SAVE_EVERY == 0:
#         save_path = config['save_model_path']
#         state = {'model_state_dict': policy_net.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
#         torch.save(state, save_path)
#         target_net.load_state_dict(policy_net.state_dict())
# env.close()

# save_path = config['final_model_path']
# state = {'model_state_dict': policy_net.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
# torch.save(state, save_path)

# episode = np.linspace(start=1, stop=num_episodes, num=num_episodes)

# plt.figure()
# plt.plot(episode, duration_history, linewidth=2.0)
# plt.xlabel('episode')
# plt.ylabel('duration')
# plt.title('Performance')

# plt.figure()
# plt.plot(episode, loss_history, linewidth=2.0)
# plt.xlabel('episode')
# plt.ylabel('loss')
# plt.title('Loss')

# plt.figure()
# plt.plot(episode, mx_history, linewidth=2.0)
# plt.xlabel('episode')
# plt.ylabel('max x')
# plt.title('Maximum X Position')

# plt.show()
