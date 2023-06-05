import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, inputs_dim=5, outputs_dim=1):
        super(DQN, self).__init__()
        self.in_dim = inputs_dim
        self.out_dim = outputs_dim

        # self.fc_1 = nn.Linear(self.in_dim, 2*self.in_dim)
        # self.bn_1 = nn.BatchNorm1d(2*self.in_dim)
        # self.fc_2 = nn.Linear(2*self.in_dim, 4*self.in_dim)
        # self.bn_2 = nn.BatchNorm1d(4*self.in_dim)
        # self.fc_3 = nn.Linear(4*self.in_dim, 2*self.in_dim)
        # self.bn_3 = nn.BatchNorm1d(2*self.in_dim)
        # self.fc_4 = nn.Linear(2*self.in_dim, self.in_dim)
        # self.bn_4 = nn.BatchNorm1d(self.in_dim)
        # self.fc_out = nn.Linear(self.in_dim, self.out_dim)
        # self.bn_out = nn.BatchNorm1d(self.out_dim)

        self.fc_1 = nn.Linear(self.in_dim, 64)
        self.bn_1 = nn.BatchNorm1d(64)
        self.fc_2 = nn.Linear(64, 128)
        self.bn_2 = nn.BatchNorm1d(128)
        self.fc_3 = nn.Linear(128, 128)
        self.bn_3 = nn.BatchNorm1d(128)
        self.fc_4 = nn.Linear(128, 8)
        self.bn_4 = nn.BatchNorm1d(8)
        self.fc_out = nn.Linear(8, self.out_dim)
        self.bn_out = nn.BatchNorm1d(self.out_dim)

        self.relu = nn.LeakyReLU()
        
    def forward(self, state, action):
        '''
        args:
            state: [B, 4]
            action: [B, 1] (B: batch size)
        return:
            Q*(s,a) : [B, 1]
        '''
        x = torch.cat((state, action), dim=1)
        a1 = self.relu(self.bn_1(self.fc_1(x)))
        a2 = self.relu(self.bn_2(self.fc_2(a1)))
        a3 = self.relu(self.bn_3(self.fc_3(a2)))
        a4 = self.relu(self.bn_4(self.fc_4(a3)))
        y = self.bn_out(self.fc_out(a4))
        return y