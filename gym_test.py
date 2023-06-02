import gym
from icecream import ic

env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(10):
#    action = policy(observation)  # User-defined policy function 
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print("...................Step...............")
    ic(observation)
    ic(reward)
    ic(terminated)
    ic(truncated)
    ic(info)
    if terminated or truncated:
        observation, info = env.reset()
env.close()