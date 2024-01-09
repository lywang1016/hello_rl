import gymnasium as gym
from network import ActorNetwork
import torch as T
from utils import Action_adapter

env = gym.make('Pendulum-v1', render_mode="human")
env._max_episode_steps = 200

action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]
max_action = float(env.action_space.high[0])
actor = ActorNetwork(action_dim, state_dim, 0.0003)
actor.load_best()

obs, info = env.reset()
done = False 
while not done:
    with T.no_grad():
        state = T.tensor(obs.reshape(1, -1), dtype=T.float).to(actor.device)
        alpha, beta = actor(state)
    mode = (alpha) / (alpha + beta)
    action = Action_adapter(mode.cpu().numpy()[0], max_action)
    obs, reward, done, truncated, info = env.step(action)
    done = done or truncated
    
    env.render()
env.close()