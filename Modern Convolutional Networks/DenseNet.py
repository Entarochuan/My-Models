"""
    Filename: Resnet.py
    Author  : Yichuan Ma
    Date :  2022/7/20
    Reference: D2L
    Description: 实现DenseNet
"""

import torch
from torch import nn
from d2l import torch as d2l
import My_training_function
import My_training_function.train as train

# 批量进行规范化，激活，卷积
def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1)
    )


# 过渡层，控制通道数
def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),  # 缩小通道数
        nn.AvgPool2d(kernel_size=2, stride=2)  # 减半高和宽
    )


# 稠密块
class DenseBlock(nn.Module):

    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels*i + input_channels, num_channels   # 每一层都把上层的输入纳入本层的考虑
            ))

        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)  # 每个网络的输出大小总为num_channels
        return X


class DenseNet:

    def __init__(self, num_channels=64, growth_rate=32, num_convs_in_blocks=[4 ,4, 4, 4]):
        self.b1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        blks = []

        for i, num_convs in enumerate(num_convs_in_blocks):  # 构建一个计数迭代器
            blks.append(DenseBlock(num_convs, num_channels, growth_rate))  # num_channels又被称为增长率
            num_channels += num_convs * growth_rate  # 在原先的基础上加上新的通道

            if i != len(num_convs_in_blocks)-1:
                blks.append(transition_block(num_channels, num_channels//2))  #通道数减半
                num_channels = num_channels//2

        self.num_channels = num_channels

        self.net = nn.Sequential(
            self.b1, *blks,
            nn.BatchNorm2d(self.num_channels) , nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.num_channels, 10)
        )

def Baseline():

    net = DenseNet().net
    lr, num_epochs, batch_size = 0.1, 10, 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    train.train_with_GPU(net, train_iter, test_iter, num_epochs, lr, device=train.try_gpu())


if __name__ == '__main__' :
    Baseline()