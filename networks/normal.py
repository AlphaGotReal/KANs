import os
import sys

import torch
import torch.nn as nn

class Normal(nn.Module):
    def __init__(self, activation, neurons: list[int]):

        super(Normal, self).__init__()

        model = []
        for n1, n2 in zip(neurons, neurons[1:]):
            model.append(nn.Linear(n1, n2, bias=True))
            model.append(activation)
        model.pop()

        self.seq = nn.Sequential(*model)

    def forward(self, x):
        return self.seq(x)
