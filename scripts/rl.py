import gym
import math
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from tqdm import tqdm
from heapq import heapify, heappop, heappush
import torch
import torch.nn as nn
from reward import reward_function_1
from network import DQN

with open('D:\python\code\hello_rl\scripts\config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

env = gym.make(config['game'])
env._max_episode_steps = 1005
obs = env.reset()
n_states = len(obs)
n_actions = env.action_space.n  # Get number of actions from gym action space

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define dataset
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Main
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
SAVE_EVERY = 5
steps_done = 0

memory = ReplayMemory(10000)

policy_net = DQN().to(device)  # Q*(s,a)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
policy_net.train()

optimizer = torch.optim.Adam(policy_net.parameters())

def select_action(state):
    global steps_done
    global n_actions
    global policy_net

    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:   
        policy_net.eval()
        state_ = torch.from_numpy(state).to(device)
        state_ = state_.view(1, state_.shape[0])    #shape:[1, 4]
        queue = []
        heapify(queue)
        for i in range(n_actions):
            a = torch.tensor(i).to(device)
            a_ = a.float().view(1, 1)
            value = policy_net(state_, a_)
            value = value.cpu().detach().numpy()[0][0]
            heappush(queue, (-value, i))
        value, a = heappop(queue)
        policy_net.train()
        return a
    else:
        return np.random.randint(n_actions)

def optimize_model():
    global steps_done
    global n_actions
    global policy_net

    if len(memory) < BATCH_SIZE:
        return 5

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state = torch.from_numpy(batch.state[0]).to(device)
    action = torch.tensor(batch.action[0]).to(device)
    reward = torch.tensor(batch.reward[0]).to(device)
    state_batch = state.view(1, state.shape[0])
    action_batch = action.view(1, 1)
    reward_batch = reward.view(1, 1)
    for i in range(1, BATCH_SIZE):
        state = torch.from_numpy(batch.state[i]).to(device)
        action = torch.tensor(batch.action[i]).to(device)
        reward = torch.tensor(batch.reward[i]).to(device)
        state_batch = torch.cat((state_batch, state.view(1, state.shape[0])), dim=0)
        action_batch = torch.cat((action_batch, action.view(1, 1)), dim=0)
        reward_batch = torch.cat((reward_batch, reward.view(1, 1)), dim=0)

    next_values = torch.zeros((BATCH_SIZE,1), device=device)
    for i in range(BATCH_SIZE):
        next_state = torch.from_numpy(batch.next_state[i]).to(device)
        next_state = next_state.view(1, next_state.shape[0])
        val_list = []
        for j in range(n_actions):
            a = torch.tensor(j).to(device)
            a = a.float().view(1, 1)
            value = target_net(next_state, a)
            value = value.cpu().detach().numpy()[0][0]
            val_list.append(value)
        max_val = max(val_list)
        next_values[i][0] = torch.tensor(max_val).to(device)
    next_values = (next_values * GAMMA) + reward_batch

    cur_values = policy_net(state_batch, action_batch)

    criterion = nn.SmoothL1Loss()
    loss = criterion(cur_values, next_values)

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
    return float(loss.cpu().detach())

# Training
num_episodes = 200
duration_history = []
loss_history = []
mx_history = []
good_cnt = 0
for i_episode in tqdm(range(num_episodes)):
    if good_cnt>20:
        num_episodes = i_episode
        break
    obs = env.reset()
    done = False 
    life_cnt = 0
    total_loss = 0  
    max_x = 0
    while not done:
        action = select_action(obs)
        new_obs, reward, done, info = env.step(action)
        if abs(new_obs[0]) > max_x:
            max_x = abs(new_obs[0])
        reward = reward_function_1(new_obs[0], new_obs[2])
        env.render()
        life_cnt += 1
        memory.push(obs, action, new_obs, reward)
        loss = optimize_model()
        total_loss += loss
        obs = new_obs
        if life_cnt > 1000:
            good_cnt += 1
            break
    if life_cnt < 1000:
        good_cnt = 0

    print(' Last episode life time is: ' + str(life_cnt))
    duration_history.append(life_cnt)
    print('Last episode loss is: ' + str(total_loss/life_cnt))
    loss_history.append(total_loss/life_cnt)
    print('Last episode max x is: ' + str(max_x))
    mx_history.append(max_x)

    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    if i_episode % SAVE_EVERY == 0:
        save_path = config['save_model_path']
        state = {'model_state_dict': target_net.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
        torch.save(state, save_path)
env.close()

save_path = config['final_model_path']
state = {'model_state_dict': policy_net.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
torch.save(state, save_path)

episode = np.linspace(start=1, stop=num_episodes, num=num_episodes)

plt.figure()
plt.plot(episode, duration_history, linewidth=2.0)
plt.xlabel('episode')
plt.ylabel('duration')
plt.title('Performance')

plt.figure()
plt.plot(episode, loss_history, linewidth=2.0)
plt.xlabel('episode')
plt.ylabel('loss')
plt.title('Loss')

plt.figure()
plt.plot(episode, mx_history, linewidth=2.0)
plt.xlabel('episode')
plt.ylabel('max x')
plt.title('Maximum X Position')

plt.show()