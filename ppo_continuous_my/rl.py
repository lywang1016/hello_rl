import gymnasium as gym
import yaml
from tqdm import tqdm
import torch as T
import torch.nn as nn
from network import ActorNetwork, CriticNetwork
from memory import Trajectory, Memory
from utils import plot_learning_curve, Action_adapter

with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

env = gym.make(config['game'])
observation, info = env.reset()
action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]

GAMMA = 0.99
LR = 1e-4
EPS = 0.2
GPI_LOOP = 50
EVALUATION_EPOCH = 5
IMPROVEMENT_EPOCH = 2
MEMORY_SIZE = 4096
BATCH_SIZE = 8
BOOTSTRAPPING = 256

actor = ActorNetwork(action_dim, state_dim, LR)
critic = CriticNetwork(state_dim, LR)
memory = Memory(BATCH_SIZE)
criterion = nn.MSELoss().to(critic.device)
score_history = []

def select_action(observation):
    state = T.tensor(observation, dtype=T.float).to(actor.device)
    alpha, beta, dist = actor(state)
    action = dist.sample()
    action = T.clamp(action, 0, 1)
    probs_old = T.squeeze(dist.log_prob(action)).item()
    return T.squeeze(action).item(), probs_old

for i in range(GPI_LOOP):
    print('---------------------- GPI Loop ' + str(i+1) + ' :' + '----------------------')

    print('Generate trajectories...')
    steps = 0
    episode_num = 0
    trajectories = []
    loop_score = []
    while steps < MEMORY_SIZE:
        observation, info = env.reset()
        trajectory = Trajectory()
        done = False
        score = 0
        while not done:
            action, probs_old = select_action(observation)
            a = Action_adapter(action, 2)
            observation_, reward, done, truncated, info  = env.step(a)
            done = done or truncated
            score += reward
            trajectory.remember(observation, action, reward, observation_, done, probs_old)
            observation = observation_
        score_history.append(score)
        loop_score.append(score)
        steps += trajectory.length
        episode_num += 1
        trajectories.append(trajectory)
    average_episode_score = sum(loop_score) / float(episode_num)
    print("\tAverage episode score: %.1f" % average_episode_score)

    print('Generate memory...')
    memory.clear_memory()
    for trajectory in trajectories:
        for i in range(trajectory.length):
            reward_sum = 0
            value_bootstrapping = 0
            discount = 1 / GAMMA
            for j in range(BOOTSTRAPPING):
                discount *= GAMMA
                if i+j < trajectory.length:
                    reward_sum += discount * trajectory.reward[i+j]
            if i+BOOTSTRAPPING < trajectory.length:
                if not trajectory.done[i+BOOTSTRAPPING]:
                    state_bootstrapping = trajectory.states[i+BOOTSTRAPPING]
                    state = T.tensor(state_bootstrapping, dtype=T.float).to(critic.device)
                    value_bootstrapping = critic(state)
                    value_bootstrapping = T.squeeze(value_bootstrapping).item()
            returns = reward_sum + discount * GAMMA * value_bootstrapping
            memory.store_memory(trajectory.states[i], trajectory.action[i], returns, trajectory.probs_old[i])

    print('Evaluation...')
    for epoch in tqdm(range(EVALUATION_EPOCH)):
        state_arr, action_arr, returns_arr, probs_old_arr, batches = memory.generate_batches()
        for batch in batches:
            if len(batch) == BATCH_SIZE:
                states = T.tensor(state_arr[batch], dtype=T.float).to(critic.device)
                returns = T.tensor(returns_arr[batch], dtype=T.float).to(critic.device)
                critic_value = critic(states)
                critic_value = T.squeeze(critic_value) 
                critic_loss = criterion(returns.view(BATCH_SIZE, 1), critic_value.view(BATCH_SIZE, 1)) 
                critic.optimizer.zero_grad()
                critic_loss.backward()
                critic.optimizer.step()

    print('Improvement...')
    for epoch in tqdm(range(IMPROVEMENT_EPOCH)):
        state_arr, action_arr, returns_arr, probs_old_arr, batches = memory.generate_batches()
        for batch in batches:
            if len(batch) == BATCH_SIZE:
                states = T.tensor(state_arr[batch], dtype=T.float).to(actor.device)
                actions = T.tensor(action_arr[batch], dtype=T.float).to(actor.device)
                returns = T.tensor(returns_arr[batch], dtype=T.float).to(actor.device)
                probs_old = T.tensor(probs_old_arr[batch], dtype=T.float).to(actor.device)
                alpha, beta, dist = actor(states)
                dist_entropy = dist.entropy()
                probs = dist.log_prob(actions)
                critic_value = critic(states)
                critic_value = T.squeeze(critic_value)
                advantage = returns - critic_value
                prob_ratio = (probs - probs_old).exp()
                weighted_probs = prob_ratio * advantage
                weighted_clipped_probs = T.clamp(prob_ratio, 1-EPS, 1+EPS) * advantage
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs) - 1e-3*dist_entropy
                actor_loss = actor_loss.mean()
                actor.optimizer.zero_grad()
                actor_loss.backward()
                actor.optimizer.step()

    actor.save_checkpoint()
    critic.save_checkpoint()

x = [i+1 for i in range(len(score_history))]
plot_learning_curve(x, score_history, config['figure_path'])
