"""
    Filename : Concise_type.py
    author   : Yichuan Ma
    date     : 2022/7/5
    reference: dive into deep learning
    Description: 使用简洁的api实现 weight decay
"""

import torch
import Data_Synthetic as Data
from My_training_function.Animator import Animator
from torch import nn
from d2l import torch as d2l

n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = Data.synthetic_data(true_w, true_b, n_train)
train_iter = Data.load_array(train_data, batch_size)
test_data = Data.synthetic_data(true_w, true_b, n_test)
test_iter = Data.load_array(test_data, batch_size)

def train_concise(lambd=0):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003

    trainer = torch.optim.SGD([
        {"params": net[0].weight, 'weight_decay': lambd},   #直接指定训练超参数
        {"params": net[0].bias}], lr=lr)
    animator = Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])

    for epoch in range(num_epochs):
        for X,y in train_iter:
            trainer.zero_grad()
            l = loss(net(X),y)
            l.mean().backward()
            trainer.step()
        if (epoch+1)%5 == 0 :
            animator.add(epoch+1, ((d2l.evaluate_loss(net, train_iter, loss),
                          d2l.evaluate_loss(net, test_iter, loss))))
    animator.show_img()

if __name__ =='__main__':
    train_concise(0)