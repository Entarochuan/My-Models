"""
    Filename: Batch_Normalization.py
    Author  : Yichuan Ma
    Date :  2022/7/20
    Reference: D2L
    Description: 批量归一化
"""

# 思路：固定小批量中的均值和方差

import torch
from torch import nn
from d2l import torch as d2l
import My_training_function.train as train

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):

    if not torch.is_grad_enabled():
        X_hat = (X-moving_mean)/torch.sqrt(moving_var)

    else:
        assert len(X.shape) in (2, 4)  #要么是全连接层，要么是卷积层
        #全连接层，在特征维度上规范化
        if len(X.shape) == 2:
            mean = X.mean(dim=0) #对每列求均值
            var = ((X-mean)**2).mean(dim=0)
        # 使用二维卷积层，计算通道维上（axis=1）的均值和方差。
        # 这里我们需要保持X的形状以便后面可以做广播运算
        else:
            mean = X.mean(dim=(0, 2, 3), keepdim=True) #按照通道数求均,通道dim=1
            var = ((X-mean)**2).mean(dim=(0, 2, 3), keepdim=True)
        #标准化
        X_hat = (X-mean) / torch.sqrt(var+eps)
        moving_mean = momentum * moving_mean + (1.0-momentum) * mean  #更新全局均值(使用加权求和的方式)
        moving_var = momentum * moving_var + (1.0-momentum) * var

    Y = gamma * X_hat + beta

    return Y, moving_mean.data, moving_var.data

class BatchNorm(nn.Module):

    def __init__(self, num_features, num_dims):
        super().__init__()

        if num_dims == 2: #全连接
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)

        #定义层参数
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):

        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)

        Y, self.moving_mean, self.moving_var = batch_norm(X, self.gamma,
                                                          self.beta, self.moving_mean,
                                                          self.moving_var, eps=1e-5, momentum=0.9)
        return Y

def Baseline():

    net = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
        nn.Linear(16 * 4 * 4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
        nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
        nn.Linear(84, 10))

    lr, num_epochs, batch_size = 1.0, 10, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    #d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

    train.train_with_GPU(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

if __name__ == '__main__':
    Baseline()