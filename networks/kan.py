import os
import sys

import torch
import torch.nn as nn

class KANFunction(nn.Module):
    def __init__(self, polynomial_degree=2):
        super(KANFunction, self).__init__()
        self.polynomial_degree = 2
        self.coeff = nn.Parameter(torch.rand((self.polynomial_degree+1, 1)))

    def forward(self, x):
        powers = torch.arange(self.polynomial_degree + 1, dtype=torch.float32)
        feat = x ** powers
        return feat @ self.coeff


class KAN(nn.Module):
    def __init__(self, n, degree=2):
        super().__init__()
        self.layer1 = nn.ModuleList([KANFunction(degree) for _ in range(n)])
        self.layer2 = nn.ModuleList([KANFunction(degree) for _ in range(n)])

    def forward(self, x):
        out = [f(g(x)) for g, f in zip(self.layer1, self.layer2)]
        out = torch.cat(out, dim=1)
        return torch.sum(out, dim=1, keepdim=True)
