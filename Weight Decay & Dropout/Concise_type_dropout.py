"""
    Filename : Concise_type_dropout.py
    author   : Yichuan Ma
    date     : 2022/7/5
    reference: dive into deep learning
    Description: api实现dropout
"""

import torch
from torch import nn
from d2l import torch as d2l
import My_training_function.Dataloader_FashionMNIST as DataLoader
import My_training_function.train as Train_funcs


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

def train():
    dropout1, dropout2 = 0.2, 0.5
    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(784, 256),
                        nn.ReLU(),
                        nn.Dropout(dropout1),
                        nn.Linear(256, 256),
                        nn.ReLU(),
                        nn.Dropout(dropout2),
                        nn.Linear(256, 10) )
    net.apply(init_weights)
    num_epochs, lr, batch_size = 10, 0.5, 256
    loss = nn.CrossEntropyLoss(reduction='none')
    train_iter, test_iter = DataLoader.FashionMnist_Dataset(batch_size=256).data_iter()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    Train_funcs.train_net_my(net, train_iter, test_iter, loss, num_epochs, trainer, my_model=False)

if __name__ =='__main__':
    train()