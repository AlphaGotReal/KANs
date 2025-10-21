import os
import sys

import torch
import torch.nn as nn

class Normal(nn.Module):
    def __init__(self, activation, n):

        super(Normal, self).__init__()
        self.lin1 = nn.Linear(1, n, bias=True)
        self.activation = activation
        self.lin2 = nn.Linear(n, 1, bias=True)

    def forward(self, x):
        return self.lin2(self.activation(self.lin1(x)))

    def intermediate(self, x):
        return self.activation(self.lin1(x))
