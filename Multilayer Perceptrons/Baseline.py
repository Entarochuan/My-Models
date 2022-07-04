"""
    Filename : Train.py
    author   : Yichuan Ma
    date     : 2022/7/4
    reference: dive into deep learning
    description: 训练流程，使用d2l定义的训练函数
"""

import torch
from torch import nn
import sys
sys.path.append('./My_training_function/')
import DataLoader
import Multilayer
import My_training_function.train as train

def process(params, net, dataset):
    loss = nn.CrossEntropyLoss(reduction='none')
    num_epochs, lr = 30, 0.0015
    train_iter, test_iter = dataset.data_iter()
    updater = torch.optim.SGD(params.Parameters, lr=lr)
    train.train_net_my(net, train_iter, test_iter, loss, num_epochs, updater, params)

def main():
    dataset = DataLoader.FashionMnist_Dataset()
    Parameters = Multilayer.Multilayer_Perceptron()
    process(Parameters, Multilayer.net, dataset)

if __name__ == "__main__":
    main()