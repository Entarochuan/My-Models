"""
    Filename:Convolutional Functions.py
    Author  :Yichuan Ma
    Date : 2022/7/15
    Reference: D2L
    Description:实现了卷积网络所需的基本函数
"""

import torch
from torch import nn
from d2l import torch as d2l

def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0]-h+1, X.shape[1]-w+1))

    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+h, j:j+w] * K).sum() #卷积求和


class Conv2D(nn.Module):  #nn库中Conv2D卷积的实现

    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))  #rand均匀分布, randn正态分布
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight)+self.bias  #返回矩阵的每个位置处加上偏置

