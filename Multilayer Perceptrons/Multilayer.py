"""
    Filename : Multilayer.py
    author   : Yichuan Ma
    date     : 2022/7/4
    reference: dive into deep learning
    description: 定义了一个一层的感知机，及其训练流程需要的函数
"""

import torch
from torch import nn
from d2l import torch as d2l

class Multilayer_Perceptron:
    def __init__(self, input=784, output=10, hidden_size=256):
        self.input, output, size = input, output, hidden_size
        self.W1 = nn.Parameter(torch.randn(
            input,hidden_size,requires_grad=True)*0.01)
        self.b1 = nn.Parameter(torch.zeros(hidden_size, requires_grad=True))
        self.W2 = nn.Parameter(torch.randn(
            hidden_size, output, requires_grad=True) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(output, requires_grad=True))
        self.Parameters = [self.W1, self.b1, self.W2, self.b2]

def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X,a)  #取值和0的最大

def net(x, parameters):
    x = x.reshape((-1, parameters.input))
    H = relu((torch.matmul(x, parameters.W1) + parameters.b1))
    return torch.matmul(H, parameters.W2) + parameters.b2

