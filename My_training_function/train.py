"""
    Filename : train.py
    author   : Yichuan Ma
    date     : 2022/7/4
    reference: dive into deep learning
    Description: 训练流程库函数
"""

import torch
from Animator import Animator
import evaluation

def train_epoch_my(net, train_iter, loss, updater, lr, parameters): #@save
    """一个迭代周期的训练流程"""
    if isinstance(net, torch.nn.Module):
        net.train() #设置为训练模式
    metric = evaluation.Accumulator(3) #三个累加变量

    for X, y in train_iter:
        y_hat = net(X, parameters)
        ls = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer): #使用内置函数
            updater.zero_grad()
            ls.mean().backward()
            updater.step()
        else:
            ls.sum().backward()
            updater(X.shape[0], parameters, lr)
    #三个累加量：loss, accuracy, 总个数
    #print(ls)
    metric.add(float(ls.sum()), evaluation.accuracy(y_hat, y), y.numel())
    #print(y.numel())
    #返回训练损失和训练精度
    return metric[0]/metric[2], metric[1]/metric[2]

def train_net_my(net, train_iter, test_iter, loss, num_epochs, updater, parameters, lr=0.01): #@save
    """训练模型"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        print("epoch",epoch+1,":")
        train_metrics = train_epoch_my(net, train_iter, loss, updater, lr, parameters) #返回一个元组
        print(train_metrics)
        test_acc = evaluation.evaluate_accuracy(net, test_iter, parameters)
        animator.add(epoch + 1, train_metrics + (test_acc,)) #绘图
    train_loss, train_acc = train_metrics
    animator.show_img()

    """当断言不成立时退出"""
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


