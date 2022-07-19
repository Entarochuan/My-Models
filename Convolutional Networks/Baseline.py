"""
    Filename:Baseline.py
    Author  :Yichuan Ma
    Date : 2022/7/15
    Reference: D2L
    Description:使用Lenet进行基本训练
"""

from Dataloader_FashionMNIST import FashionMnist_Dataset as dataset
import Lenet
import My_training_function.train as train
import torch

def Baseline(cuda=True, lr=0.9, epoch=10):

    train_iter, test_iter = dataset().data_iter()
    net = Lenet.Lenet().net
    if cuda :
        device = train.try_gpu()
    else :
        device = torch.device('cpu')
    train.train_with_GPU(net, train_iter, test_iter, epoch, lr, device)

if __name__ == '__main__':
    Baseline()