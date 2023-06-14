import gymnasium as gym
import yaml
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

print("TEST")

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

env = gym.make(config['game'], render_mode="human")
env._max_episode_steps = 10000

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
q_net = DQN(n_observations, n_actions).to(device)
checkpoint = torch.load(config['final_model_path'])
q_net.load_state_dict(checkpoint['model_state_dict'])
q_net = q_net.eval()

duration_history = []
num_episodes = 2
for i_episode in tqdm(range(num_episodes)):
    obs, info = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    done = False 
    life_cnt = 0
    while not done:
        action = q_net(obs).max(1)[1].view(1, 1)
        obs, reward, done, truncated, info = env.step(action.item())
        obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
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