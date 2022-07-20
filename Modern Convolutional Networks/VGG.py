"""
    Filename:VGG.py
    Author  :Yichuan Ma
    Date : 2022/7/19
    Reference: D2L
    Description:实现VGG
"""

import torch
from torch import nn
from d2l import torch as d2l
import My_training_function.train as train

def VGG_block(num_convs, in_channels, out_channels):  #自动生成卷积块
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)  #返回一个可以直接使用的模块

def VGG(conv_arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))):   #允许按照输入结构生成对应网络,输入结构代表输入输出通道
    conv_blks = []
    in_channels = 1

    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(VGG_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(*conv_blks, nn.Flatten(),
                         nn.Linear(out_channels*7*7, 4096), nn.ReLU(), nn.Dropout(0.5),
                         nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
                         nn.Linear(4096, 10))

def test():
    net = VGG()
    X = torch.randn(size=(1, 1, 224, 224))
    for blk in net:
        X = blk(X)
        print(blk.__class__.__name__, 'output shape:\t', X.shape)

def Baseline():
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    ratio = 32
    smaller_conv = [(pair[0], pair[1] // ratio) for pair in conv_arch]
    net = VGG(smaller_conv)
    lr, num_epochs, batch_size = 0.05, 10, 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    train.train_with_GPU(net, train_iter, test_iter, num_epochs, lr, train.try_gpu())

if __name__ == '__main__':
    Baseline()