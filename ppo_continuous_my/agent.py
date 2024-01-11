import torch as T
import copy
import math
import numpy as np
from network import ActorNetwork, CriticNetwork

class Agent(object):
    def __init__(self, **kwargs):
        # Init hyperparameters for PPO agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
        self.__dict__.update(kwargs)
        # Build Actor Critic
        self.actor = ActorNetwork(self.action_dim, self.state_dim, self.a_lr)
        self.critic = CriticNetwork(self.state_dim, self.c_lr)
        # Build Trajectory holder
        self.s_hoder = np.zeros((self.T_horizon, self.state_dim),dtype=np.float32)
        self.a_hoder = np.zeros((self.T_horizon, self.action_dim),dtype=np.float32)
        self.r_hoder = np.zeros((self.T_horizon, 1),dtype=np.float32)
        self.s_next_hoder = np.zeros((self.T_horizon, self.state_dim),dtype=np.float32)
        self.logprob_a_hoder = np.zeros((self.T_horizon, self.action_dim),dtype=np.float32)
        self.done_hoder = np.zeros((self.T_horizon, 1),dtype=np.bool_)
        self.dw_hoder = np.zeros((self.T_horizon, 1),dtype=np.bool_)
	
    def stochastic_action(self, observation):
        with T.no_grad():
            state = T.tensor(observation.reshape(1, -1), dtype=T.float).to(self.actor.device)
            dist = self.actor.get_dist(state)
            action = dist.sample()
            action = T.clamp(action, 0, 1)
            probs_old = dist.log_prob(action).cpu().numpy().flatten()
            return action.cpu().numpy()[0], probs_old
    
    def deterministic_action(self, observation):
        with T.no_grad():
            state = T.tensor(observation.reshape(1, -1), dtype=T.float).to(self.actor.device)
            action = self.actor.deterministic_act(state)
            return action.cpu().numpy()[0]
        
    def load_checkpoints(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def save_checkpoints(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_best(self):
        self.actor.load_best()
        self.critic.load_best()

    def save_best(self):
        self.actor.save_best()
        self.critic.save_best()

    def train(self):
        self.entropy_coef*=self.entropy_coef_decay

        '''Prepare PyTorch data from Numpy data'''
        s = T.from_numpy(self.s_hoder).to(self.actor.device)
        a = T.from_numpy(self.a_hoder).to(self.actor.device)
        r = T.from_numpy(self.r_hoder).to(self.actor.device)
        s_next = T.from_numpy(self.s_next_hoder).to(self.actor.device)
        logprob_a = T.from_numpy(self.logprob_a_hoder).to(self.actor.device)
        done = T.from_numpy(self.done_hoder).to(self.actor.device)
        dw = T.from_numpy(self.dw_hoder).to(self.actor.device)
        
        ''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
        with T.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_next)
            
            '''dw for TD_target and Adv'''
            deltas = r + self.gamma * vs_ * (~dw) - vs
            deltas = deltas.cpu().flatten().numpy()
            adv = [0]
            
            '''done for GAE'''
            for dlt, mask in zip(deltas[::-1], done.cpu().flatten().numpy()[::-1]):
                advantage = dlt + self.gamma * self.lambd * adv[-1] * (~mask)
                adv.append(advantage)
            adv.reverse()
            adv = copy.deepcopy(adv[0:-1])
            adv = T.tensor(adv).unsqueeze(1).float().to(self.actor.device)
            td_target = adv + vs
            adv = (adv - adv.mean()) / ((adv.std()+1e-4))  #sometimes helps
            
        """Slice long trajectopy into short trajectory and perform mini-batch PPO update"""
        a_optim_iter_num = int(math.ceil(s.shape[0] / self.a_optim_batch_size))
        c_optim_iter_num = int(math.ceil(s.shape[0] / self.c_optim_batch_size))
        
        for i in range(self.K_epochs):
            #Shuffle the trajectory, Good for training
            perm = np.arange(s.shape[0])
            np.random.shuffle(perm)
            perm = T.LongTensor(perm).to(self.actor.device)
            s, a, td_target, adv, logprob_a = \
				s[perm].clone(), a[perm].clone(), td_target[perm].clone(), adv[perm].clone(), logprob_a[perm].clone()
            
            '''update the actor'''
            for i in range(a_optim_iter_num):
                index = slice(i * self.a_optim_batch_size, min((i + 1) * self.a_optim_batch_size, s.shape[0]))
                distribution = self.actor.get_dist(s[index])
                dist_entropy = distribution.entropy().sum(1, keepdim=True)
                logprob_a_now = distribution.log_prob(a[index])
                ratio = T.exp(logprob_a_now.sum(1,keepdim=True) - logprob_a[index].sum(1,keepdim=True))  # a/b == exp(log(a)-log(b))
                surr1 = ratio * adv[index]
                surr2 = T.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index]
                a_loss = -T.min(surr1, surr2) - self.entropy_coef * dist_entropy
                
                self.actor.optimizer.zero_grad()
                a_loss.mean().backward()
                T.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.actor.optimizer.step()
                
            '''update the critic'''
            for i in range(c_optim_iter_num):
                index = slice(i * self.c_optim_batch_size, min((i + 1) * self.c_optim_batch_size, s.shape[0]))
                c_loss = (self.critic(s[index]) - td_target[index]).pow(2).mean()
                for name,param in self.critic.named_parameters():
                    if 'weight' in name:
                        c_loss += param.pow(2).sum() * self.l2_reg

                self.critic.optimizer.zero_grad()
                c_loss.backward()
                self.critic.optimizer.step()
                
    def put_data(self, s, a, r, s_next, logprob_a, done, dw, idx):
        self.s_hoder[idx] = s
        self.a_hoder[idx] = a
        self.r_hoder[idx] = r
        self.s_next_hoder[idx] = s_next
        self.logprob_a_hoder[idx] = logprob_a
        self.done_hoder[idx] = done
        self.dw_hoder[idx] = dw