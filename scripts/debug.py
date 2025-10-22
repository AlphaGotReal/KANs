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

model = KAN(2)
