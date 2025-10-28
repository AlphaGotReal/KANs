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

from networks.normal import Normal

activation = nn.ReLU()
model = Normal(activation, [1, 64, 64, 64, 64, 1])
print(model)
loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# dataset
X = torch.tensor(np.linspace(-3.14, 3.14, 1000), dtype=torch.float32)
X = X.reshape(X.shape + (1,))
Y = nn.ReLU()(torch.sin(X * 2 * np.pi)) - torch.cos(X)

def plot():
    global X, Y, model
    from matplotlib import pyplot as plt
    y, pred_y, x = Y.numpy(), model(X).detach().numpy(), X.numpy()
    plt.clf()
    plt.plot(x, y)
    plt.plot(x, pred_y)
    plt.pause(0.001)

for epoch in range(10000):
    pred_Y = model(X)
    l = loss(pred_Y, Y)
    optimizer.zero_grad()
    l.backward()
    optimizer.step()
    print(f"epoch={epoch} \tloss={l}\t\t", end='\r', flush=True)
    plot()

plot()
