import gym
import csv
import yaml
from tqdm import tqdm
from reward import reward_function_1

with open('D:\workspace\hello_rl\scripts\config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

f = open(config['pretrain_data_path'], 'w', newline='')
writer = csv.writer(f)
env = gym.make(config['game'])

num_episodes = 300
for i_episode in tqdm(range(num_episodes)):
    obs, info = env.reset()
    done = False 
    while not done:
        random_action = env.action_space.sample()
        # new_obs, reward, done, info = env.step(random_action)
        new_obs, reward, done, truncated, info = env.step(random_action)
        reward = reward_function_1(new_obs[1], new_obs[2])
        env.render()

        row = []
        for state in obs:
            row.append(state)
        row.append(random_action)
        row.append(reward)
        for state in new_obs:
            row.append(state)
        writer.writerow(row)

        obs = new_obs

env.close()
f.close()