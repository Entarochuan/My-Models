"""
    Filename:Alexnet.py
    Author  :Yichuan Ma
    Date : 2022/7/19
    Reference: D2L
    Description:实现了Alexnet
"""

import torch
from torch import nn
from d2l import torch as d2l
import My_training_function.train as train

class Alexnet:

    def __init__(self):
        # 使用一个11*11的更大窗口来捕捉对象。
        # 步幅为4，以减少输出的高度和宽度。
        # 另外，输出通道的数目远大于LeNet
        self.net = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(6400, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 10)
        )

    def Display(self, x=None):
        if x:
            X = x
        else:
            X = torch.randn(1, 1, 224, 224)

        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape: \t', X.shape)

def Baseline(cuda=True, lr=0.01, epoch=10):
    net = Alexnet().net
    batch_size = 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    lr, num_epochs = 0.01, 10
    train.train_with_GPU(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

def test():
    net = Alexnet()
    net.Display()

if __name__=='__main__':
    Baseline()
