import gymnasium as gym
from network import ActorNetwork
import torch as T

env = gym.make('Pendulum-v1', render_mode="human")
env._max_episode_steps = 10000

action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]
actor = ActorNetwork(action_dim, state_dim, 0.0003)
actor.load_checkpoint()

obs, info = env.reset()
done = False 
while not done:
    state = T.tensor(obs, dtype=T.float).to(actor.device).unsqueeze(0)
    mu, dist = actor(state)
    action = T.squeeze(mu).item()
    obs, reward, done, truncated, info = env.step([action])
    done = done or truncated
    
    env.render()
env.close()