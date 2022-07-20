"""
    Filename:Nin.py
    Author  :Yichuan Ma
    Date : 2022/7/15
    Reference: D2L
    Description: Nets in Net
"""

#NiN的结构

import torch
from torch import nn
from d2l import torch as d2l
import My_training_function.train as train

def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
    )

class NiN_net:

    def __init__(self):
        self.net = nn.Sequential(
            nin_block(1, 96, kernel_size=11, strides=4, padding=0),
            nn.MaxPool2d(3, stride=2),
            nin_block(96, 256, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            nin_block(256, 384, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.5),
            # 标签类别数是10
            nin_block(384, 10, kernel_size=3, strides=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            # 将四维的输出转成二维的输出，其形状为(批量大小,10)
            nn.Flatten()
        )

    def test(self):
        X = torch.rand(size=(1, 1, 224, 224))
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)


def Baseline():
    net = NiN_net().net
    lr, num_epochs, batch_size = 0.1, 10, 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    train.train_with_GPU(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())


if __name__ == '__main__':
    Baseline()