"""
    Filename:Lenet.py
    Author  :Yichuan Ma
    Date : 2022/7/15
    Reference: D2L
    Description:实现了Lenet的基本架构
"""

import torch
from torch import nn
from d2l import torch as d2l

class Lenet:

    def __init__(self):
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def Display(self, x=None):
        if x :
            X = x
        else:
            X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)

        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape: \t', X.shape)

def test():
    Lenet = Lenet()
    Lenet.Display()

if __name__=='__main__':
    test()
