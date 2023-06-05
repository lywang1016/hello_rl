# import gym
# from icecream import ic

# # env = gym.make("LunarLander-v2", render_mode="human")
# env = gym.make("CartPole-v1", render_mode="human")
# observation, info = env.reset(seed=42)
# for _ in range(1000):
# #    action = policy(observation)  # User-defined policy function 
#     action = env.action_space.sample()
#     observation, reward, terminated, truncated, info = env.step(action)
#     print("...................Step...............")
#     ic(observation)
#     ic(reward)
#     ic(terminated)
#     ic(truncated)
#     ic(info)
#     if terminated or truncated:
#         observation, info = env.reset()
# env.close()

import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1", render_mode="human")
n_actions = env.action_space.n
state, info = env.reset()
num_episodes = 1
rewards = []
for i_episode in range(num_episodes):
    # state, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)
        done = terminated or truncated

print("reward max:" + str(max(rewards)))
print("reward min:" + str(min(rewards)))
print(rewards)
plt.figure()
plt.plot(rewards, linewidth=2.0)
plt.show()