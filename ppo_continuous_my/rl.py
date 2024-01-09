import gymnasium as gym
import yaml
import argparse
import os
from tqdm import tqdm
import torch as T
import torch.nn as nn
from agent import Agent
from utils import plot_learning_curve, Action_adapter

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--T_horizon', type=int, default=2048, help='lenth of long trajectory')
parser.add_argument('--Max_train_steps', type=int, default=int(5e5), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(5e4), help='Model saving interval, in steps.')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--lambd', type=float, default=0.95, help='GAE Factor')
parser.add_argument('--clip_rate', type=float, default=0.2, help='PPO Clip rate')
parser.add_argument('--K_epochs', type=int, default=10, help='PPO update times')
parser.add_argument('--a_lr', type=float, default=2e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=2e-4, help='Learning rate of critic')
parser.add_argument('--l2_reg', type=float, default=1e-3, help='L2 regulization coefficient for Critic')
parser.add_argument('--a_optim_batch_size', type=int, default=64, help='lenth of sliced trajectory of actor')
parser.add_argument('--c_optim_batch_size', type=int, default=64, help='lenth of sliced trajectory of critic')
parser.add_argument('--entropy_coef', type=float, default=1e-3, help='Entropy coefficient of Actor')
parser.add_argument('--entropy_coef_decay', type=float, default=0.99, help='Decay rate of entropy_coef')
opt = parser.parse_args()

if not os.path.exists('model'): 
    os.mkdir('model')
if not os.path.exists('plots'): 
    os.mkdir('plots')

with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

env = gym.make('Pendulum-v1')
opt.state_dim = env.observation_space.shape[0]
opt.action_dim = env.action_space.shape[0]
opt.max_action = float(env.action_space.high[0])
opt.max_steps = env._max_episode_steps

env_seed = opt.seed
T.manual_seed(opt.seed)
T.cuda.manual_seed(opt.seed)
T.backends.cudnn.deterministic = True
T.backends.cudnn.benchmark = False

agent = Agent(**vars(opt)) # transfer opt to dictionary, and use it to init PPO_agent

score_history = []
traj_lenth= 0
total_steps = 0
episode = 0
best_score = -10000
while total_steps < opt.Max_train_steps:
    s, info = env.reset(seed=env_seed) # Do not use opt.seed directly, or it can overfit to opt.seed
    env_seed += 1
    done = False
    score = 0

    '''Interact & trian'''
    while not done:
        '''Interact with Env'''
        a, logprob_a = agent.stochastic_action(s) # use stochastic when training
        act = Action_adapter(a,opt.max_action) #[0,1] to [-max,max]
        s_next, r, dw, tr, info = env.step(act) # dw: dead&win; tr: truncated
        done = (dw or tr)
        score += r
        
        '''Store the current transition'''
        agent.put_data(s, a, r, s_next, logprob_a, done, dw, idx = traj_lenth)
        s = s_next
        traj_lenth += 1
        total_steps += 1

        '''Update if its time'''
        if traj_lenth % opt.T_horizon == 0:
            agent.train()
            traj_lenth = 0

        '''Save model'''
        if total_steps % opt.save_interval == 0:
            agent.save_checkpoints()
            print("Save Checkpoints Model")

    score_history.append(score)
    episode += 1
    if episode % 100 == 0:
        temp_score_history = score_history[(episode-10) : episode]
        ave_score = sum(temp_score_history) / 10
        print("Step " + str(total_steps) + ' of ' + str(opt.Max_train_steps) \
               +' last 10 episode average score: ' + str(ave_score))
        if ave_score > best_score:
            best_score = ave_score
            agent.save_best()
            print("Save Best Model")
    

x = [i+1 for i in range(len(score_history))]
plot_learning_curve(x, score_history, config['figure_path'])
