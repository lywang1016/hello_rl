import gym
import yaml
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from heapq import heapify, heappop, heappush
from network import DQN

cwd = os.getcwd()
with open(cwd+'/config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

env = gym.make(config['game'])
env._max_episode_steps = 10000
n_actions = env.action_space.n  # Get number of actions from gym action space

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
q_net = DQN().to(device)  # Q*(s,a)
checkpoint = torch.load(config['final_model_path'])
q_net.load_state_dict(checkpoint['model_state_dict'])
q_net = q_net.eval()

def select_action(state):
    global q_net

    state_ = torch.from_numpy(state).to(device)
    state_ = state_.view(1, state_.shape[0])    #shape:[1, 4]
    queue = []
    heapify(queue)
    for i in range(n_actions):
        a = torch.tensor(i).to(device)
        a_ = a.float().view(1, 1)
        value = q_net(state_, a_)
        value = value.cpu().detach().numpy()[0][0]
        heappush(queue, (-value, i))
    value, a = heappop(queue)
    return a

duration_history = []
num_episodes = 20
for i_episode in tqdm(range(num_episodes)):
    obs = env.reset()
    done = False 
    life_cnt = 0
    while not done:
        action = select_action(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        life_cnt += 1
    duration_history.append(life_cnt)
env.close()

episode = np.linspace(start=1, stop=num_episodes, num=num_episodes)

plt.figure()
plt.plot(episode, duration_history, linewidth=2.0)
plt.xlabel('episode')
plt.ylabel('duration')
plt.title('Performance')
plt.show()