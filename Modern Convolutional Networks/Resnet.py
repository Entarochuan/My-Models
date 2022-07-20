"""
    Filename: Resnet.py
    Author  : Yichuan Ma
    Date :  2022/7/20
    Reference: D2L
    Description: 实现Resnet
"""

import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import My_training_function.train as train

class Residual(nn.Module):

    def __init__(self, input_channels, num_channels,      #当前后通道不一致时需要将输出先通过1x1卷积层变形
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

        #归一化模块
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))

        if self.conv3:
            X = self.conv3(X)
        Y += X

        return F.relu(Y)

def Resnet_Block(input_channels, num_channels, num_residuals, frist_block=False):
    blk = []

    for i in range(num_residuals):
        if i==0 and not frist_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))

        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

class Resnet:

    def __init__(self):
        self.b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, padding=3, stride=2),
                                nn.BatchNorm2d(64), nn.ReLU(),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b2 = nn.Sequential(*Resnet_Block(64, 64, 2, frist_block=True))
        self.b3 = nn.Sequential(*Resnet_Block(64, 128, 2))
        self.b4 = nn.Sequential(*Resnet_Block(128, 256, 2))
        self.b5 = nn.Sequential(*Resnet_Block(256, 512, 2))

        self.net = nn.Sequential(self.b1, self.b2, self.b3, self.b4, self.b5,
                                 nn.AdaptiveAvgPool2d((1,1)),
                                 nn.Flatten(), nn.Linear(512, 10))

    def Show_net(self):
        X = torch.rand(size=(1, 1, 224, 224))
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)
    
def test():
    net = Resnet()
    net.Show_net()

def Baseline():
    net = Resnet().net
    lr, num_epochs, batch_size = 0.05, 10, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    #d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    train.train_with_GPU(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

if __name__ == '__main__':
    #test()
    Baseline()