import gymnasium as gym
import math
import yaml
import numpy as np
from tqdm import tqdm
import torch as T
from network import ActorNetwork, CriticNetwork
from memory import Trajectory, VanillaMemory
from loss import MyLoss

with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

env = gym.make(config['game'])
observation, info = env.reset()
n_actions = env.action_space.n
input_dims = env.observation_space.shape

GAMMA = 0.99
LR = 1e-4
GPI_LOOP = 20
EVALUATION_EPOCH = 1
IMPROVEMENT_EPOCH = 1
MEMORY_SIZE = 4096
BATCH_SIZE = 8
BOOTSTRAPPING = 256
FAKESTATE = np.array([c for c in range(input_dims[0])])

actor = ActorNetwork(n_actions, input_dims, LR)
critic = CriticNetwork(input_dims, LR)
vanilla_memory = VanillaMemory(BATCH_SIZE)
criterion = MyLoss().to(critic.device)

def select_action(observation):
    state = T.tensor(observation, dtype=T.float).to(actor.device)
    dist = actor(state)
    action = dist.sample()
    return T.squeeze(action).item()

for i in range(GPI_LOOP):
    print('---------------------- GPI Loop ' + str(i+1) + ' :' + '----------------------')

    print('Generate trajectories...')
    steps = 0
    episode_num = 0
    trajectories = []
    while steps < MEMORY_SIZE:
        observation, info = env.reset()
        trajectory = Trajectory()
        done = False
        while not done:
            action =  select_action(observation)
            observation_, reward, done, truncated, info  = env.step(action)
            if done:
                observation_ = FAKESTATE
            trajectory.remember(observation, action, reward, observation_)
            observation = observation_
        steps += trajectory.length
        episode_num += 1
        trajectories.append(trajectory)
    average_episode_step = steps / float(episode_num)
    print("\tAverage episode step count: %.1f" % average_episode_step)
    if average_episode_step > MEMORY_SIZE:
        actor.save_checkpoint()
        critic.save_checkpoint()
        break

    print('Generate memory...')
    vanilla_memory.clear_memory()
    for trajectory in trajectories:
        for i in range(trajectory.length):
            reward_sum = 0
            value_bootstrapping = 0
            for j in range(BOOTSTRAPPING):
                if i+j < trajectory.length:
                    reward_sum += math.pow(GAMMA, j) * trajectory.reward[i+j]
            if i+BOOTSTRAPPING < trajectory.length:
                state_bootstrapping = trajectory.states[i+BOOTSTRAPPING]
                if not (state_bootstrapping == FAKESTATE).all():
                    state = T.tensor(state_bootstrapping, dtype=T.float).to(critic.device)
                    value_bootstrapping = critic(state)
                    value_bootstrapping = T.squeeze(value_bootstrapping).item()
            returns = reward_sum + math.pow(GAMMA, BOOTSTRAPPING) * value_bootstrapping
            vanilla_memory.store_memory(trajectory.states[i], trajectory.action[i], returns)

    print('Evaluation...')
    # loss_his = []
    for epoch in tqdm(range(EVALUATION_EPOCH)):
        state_arr, action_arr, returns_arr, batches = vanilla_memory.generate_batches()
        # epoch_ave_loss = 0
        # num_batch = 0
        for batch in batches:
            if len(batch) == BATCH_SIZE:
                # num_batch += 1
                states = T.tensor(state_arr[batch], dtype=T.float).to(critic.device)
                returns = T.tensor(returns_arr[batch], dtype=T.float).to(critic.device)
                critic_value = critic(states)
                critic_value = T.squeeze(critic_value) 
                critic_loss = criterion(returns.view(BATCH_SIZE, 1), critic_value.view(BATCH_SIZE, 1))      
                # epoch_ave_loss += float(critic_loss)
                critic.optimizer.zero_grad()
                critic_loss.backward()
                critic.optimizer.step()
    #     epoch_ave_loss /= num_batch
    #     loss_his.append(epoch_ave_loss)
    # print(loss_his)

    print('Improvement...')
    # loss_his = []
    for epoch in tqdm(range(IMPROVEMENT_EPOCH)):
        state_arr, action_arr, returns_arr, batches = vanilla_memory.generate_batches()
        # epoch_ave_loss = 0
        # num_batch = 0
        for batch in batches:
            if len(batch) == BATCH_SIZE:
                # num_batch += 1
                states = T.tensor(state_arr[batch], dtype=T.float).to(actor.device)
                actions = T.tensor(action_arr[batch], dtype=T.float).to(actor.device)
                returns = T.tensor(returns_arr[batch], dtype=T.float).to(actor.device)
                dist = actor(states)
                probs = dist.log_prob(actions)
                critic_value = critic(states)
                critic_value = T.squeeze(critic_value)
                advantage = returns - critic_value
                actor_loss = -probs * advantage
                actor_loss = actor_loss.mean()
                # epoch_ave_loss += float(actor_loss)
                actor.optimizer.zero_grad()
                actor_loss.backward()
                actor.optimizer.step()
    #     epoch_ave_loss /= num_batch
    #     loss_his.append(epoch_ave_loss)
    # print(loss_his)

    actor.save_checkpoint()
    critic.save_checkpoint()
