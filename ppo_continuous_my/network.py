import os
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Beta, Normal

class ActorNetwork(nn.Module):
    def __init__(self, action_dim, state_dim, a_lr):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(os.getcwd(), 'model', 'actor_checkpoint.pth')
        self.best_file = os.path.join(os.getcwd(), 'model', 'actor_best.pth')

        self.actor = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
        )
        self.alpha_head = nn.Linear(256, action_dim)
        self.beta_head = nn.Linear(256, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=a_lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.actor(state)
        alpha = F.softplus(self.alpha_head(x)) + 1.0
        beta = F.softplus(self.beta_head(x)) + 1.0
        return alpha, beta
    
    def get_dist(self, state):
        alpha, beta = self.forward(state)
        dist = Beta(alpha, beta)
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def save_best(self):
        T.save(self.state_dict(), self.best_file)

    def load_best(self):
        self.load_state_dict(T.load(self.best_file))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, c_lr):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(os.getcwd(), 'model', 'critic_checkpoint.pth')
        self.best_file = os.path.join(os.getcwd(), 'model', 'critic_best.pth')

        self.critic = nn.Sequential(
                nn.Linear(input_dims, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=c_lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def save_best(self):
        T.save(self.state_dict(), self.best_file)

    def load_best(self):
        self.load_state_dict(T.load(self.best_file))