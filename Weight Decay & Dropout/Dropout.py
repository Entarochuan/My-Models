"""
    Filename : Dropout.py
    author   : Yichuan Ma
    date     : 2022/7/5
    reference: dive into deep learning
    Description: 从零实现暂退法,包括暂退实现和网络定义
"""

import torch
from torch import nn
from d2l import torch as d2l
import My_training_function.Dataloader_FashionMNIST as DataLoader
import My_training_function.train as Train_funcs


def dropout_decay(X,dropout): #以dropout的概率丢弃张量X中的元素
    assert 0 <= dropout <= 1

    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    mask = (torch.randn(X.shape)>dropout).float() #只保留大于阈值的部分
    return mask*X / (1.0-dropout)

class Net(nn.Module):    #网络搭建

    def __init__(self, num_inputs, num_outputs, num_hidden1, num_hidden2
                 , dp1, dp2, is_Train=True):
        super(Net, self).__init__() #继承父类的初始化
        self.num_inputs = num_inputs
        self.training = is_Train
        self.lin1 = nn.Linear(num_inputs, num_hidden1)
        self.lin2 = nn.Linear(num_hidden1,num_hidden2)
        self.lin3 = nn.Linear(num_hidden2,num_outputs)
        self.relu = nn.ReLU()
        self.dp1 = dp1
        self.dp2 = dp2

    def forward(self, X):
        out_1 = self.relu(self.lin1(X.reshape((-1,self.num_inputs))))

        if self.training :
            out = dropout_decay(out_1, self.dp1)

        out_2 = self.relu(self.lin2(out_1))

        if self.training :
            out_2 = dropout_decay(out_2, self.dp2)

        out_3 = self.lin3(out_2)
        return out_3

def train(): #使用的仍然是Fashion_Mnist数据集
    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
    net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2, 0.2, 0.5)

    num_epochs, lr, batch_size = 10, 0.5, 256
    loss = nn.CrossEntropyLoss(reduction='none')
    train_iter, test_iter = DataLoader.FashionMnist_Dataset(batch_size=256).data_iter()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    Train_funcs.train_net_my(net, train_iter, test_iter, loss, num_epochs, trainer, my_model=False)

if __name__=='__main__':
    train()
