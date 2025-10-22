#! /usr/bin/env python

import os
import sys

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURR_DIR)
sys.path.append(ROOT_DIR)

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from networks.kan import KANFunction, KAN

model = KAN(10, 6)
loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

X = torch.tensor(np.linspace(-3.14, 3.14, 1000), dtype=torch.float32)
X = X.reshape(X.shape + (1,))
Y = nn.ReLU()(torch.sin(X)) - torch.cos(X)

for epoch in range(10000):
    pred_Y = model(X)
    l = loss(pred_Y, Y)
    optimizer.zero_grad()
    l.backward()
    optimizer.step()
    print(f"loss={l}\t\t", end='\r', flush=True)

def plot():
    global X, Y, model
    from matplotlib import pyplot as plt
    y, pred_y, x = Y.numpy(), model(X).detach().numpy(), X.numpy()
    plt.plot(x, y)
    plt.plot(x, pred_y)
    plt.show()

plot()
