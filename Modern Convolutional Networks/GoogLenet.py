"""
    Filename: GoogLenet.py
    Author  : Yichuan Ma
    Date :  2022/7/20
    Reference: D2L
    Description: 包含并行连结的网络
"""

import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import My_training_function.train as train

#主要思想是通过多个尺寸的滤波器处理图片

class Inception(nn.Module):

    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):  #c1,c2,c3,c4是每条路径的输出通道数
        super(Inception, self).__init__(**kwargs)

        #线路1, 1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)

        #线路2, 3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)

        #线路3, 5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)

        #线路4, 3x3最大汇聚层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(F.relu(self.p4_1(x))))

        return torch.cat((p1, p2, p3, p4), dim=1)

class GoogLenet:

    def __init__(self):
        self.B1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.B2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                                nn.ReLU(),
                                nn.Conv2d(64, 192, kernel_size=3, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.B3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.B4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.B5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   nn.AdaptiveAvgPool2d((1,1)),
                   nn.Flatten())
        self.net = nn.Sequential(self.B1, self.B2, self.B3, self.B4, self.B5, nn.Linear(1024, 10))

    def Show_net(self):
        X = torch.rand(size=(1, 1, 96, 96))
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)

def test():
    net = GoogLenet()
    net.Show_net()

def Baseline():
    net = GoogLenet().net
    lr, num_epochs, batch_size = 0.1, 10, 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    # d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    train.train_with_GPU(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

if __name__ == '__main__':
    #test()
    Baseline()