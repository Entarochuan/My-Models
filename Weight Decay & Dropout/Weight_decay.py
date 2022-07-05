"""
    Filename : Weight_decay.py
    author   : Yichuan Ma
    date     : 2022/7/5
    reference: dive into deep learning
    Description: L2范式的权重衰减实现
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

def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2

def linreg(X,w,b):  #线性forward
    return torch.matmul(X,w)+b

def squard_loss(y_hat,y): #均方误差,y_hat和y对齐
    return (y_hat-y.reshape(y_hat.shape))**2 /2

def train(lambd=0, num_epochs=100, lr=0.03):
    w, b = init_params()
    net = lambda x : linreg(x, w, b)
    loss = squard_loss
    animator = Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])

    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y)+lambd*l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w,b],lr,batch_size)
        if (epoch + 1) % 5 == 0: #每五个作一次图
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数是：', torch.norm(w).item())
    animator.show_img()

if __name__ =='__main__':
    train(lambd=3)
