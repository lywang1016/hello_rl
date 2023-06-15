import torch.nn as nn

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.my_loss = nn.MSELoss()
        # self.my_loss = nn.CrossEntropyLoss()

    def forward(self, ref_p, est_p):
        loss = self.my_loss(ref_p, est_p)
        return loss