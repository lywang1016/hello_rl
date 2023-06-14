import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from network import ActorNetwork
import torch as T

env = gym.make('CartPole-v1', render_mode="human")
env._max_episode_steps = 10000

n_actions = env.action_space.n
input_dims=env.observation_space.shape
alpha = 0.0003
actor = ActorNetwork(n_actions, input_dims, alpha)
actor.load_checkpoint()

duration_history = []
num_episodes = 2
for i_episode in tqdm(range(num_episodes)):
    obs, info = env.reset()
    done = False 
    life_cnt = 0
    while not done:
        state = T.tensor(obs, dtype=T.float).to(actor.device)
        dist = actor(state)
        action = dist.sample()
        action = T.squeeze(action).item()
        obs, reward, done, truncated, info = env.step(action)
        
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