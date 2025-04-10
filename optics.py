import torch
import torch.nn as nn
import numpy as np


def hadamard(n):
    if n == 1:
        return np.array([[1]])
    else:
        h = hadamard(n // 2)
        return np.block([[h, h], [h, -h]])


class OpticsSPC(nn.Module):
    def __init__(self, input_size: tuple, snapshots, matrix):
        super(OpticsSPC, self).__init__()
        _, self.M, self.N = input_size
        self.snapshots = snapshots
   
        ca = torch.tensor(matrix).float()
        ca = ca.view(self.snapshots, self.M, self.N)
        self.cas = ca


    def forward(self, x):
        y = self.forward_pass(x)
        x = self.transpose_pass(y)
        return x

    def forward_pass(self, x):
        ca = self.get_coded_aperture().to(x.device)
        y = x * ca
        y = torch.sum(y, dim=(-2, -1))
        y = y.unsqueeze(-1).unsqueeze(-1)

        return y

    def transpose_pass(self, y):
        ca = self.get_coded_aperture().to(y.device)
        x = y * ca
        x = torch.sum(x, dim=1)
        x = x.unsqueeze(1)
        x = x / torch.max(x)
        return x

    def get_coded_aperture(self):
        ca = self.cas.unsqueeze(0)
        return ca
