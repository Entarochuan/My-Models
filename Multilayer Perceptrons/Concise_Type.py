"""
    Filename : Concise_Type.py
    author   : Yichuan Ma
    date     : 2022/7/4
    reference: dive into deep learning
    description: 多层感知机应用torch api的简洁实现
"""

import torch
from torch import nn
from d2l import torch as d2l
import My_training_function.DataLoader as DataLoader
import My_training_function.train as train



def init_weights(m):   #对模型的线性层作初始化
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


# .apply对每一层应用
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784,256),
                    nn.ReLU(),
                    nn.Linear(256,10))
net.apply(init_weights)
batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)
train_iter, test_iter = DataLoader.FashionMnist_Dataset(batch_size=batch_size).data_iter()

train.train_net_my(net,train_iter,test_iter,loss,num_epochs,trainer,net.parameters(),lr, False)

