"""
    Filename : train.py
    author   : Yichuan Ma
    date     : 2022/7/4, 2022/7/17, 2022/7/20
    reference: dive into deep learning
    Description: 训练流程库函数,分别定义了cpu, gpu训练流程
"""

import torch
from Animator import Animator
import evaluation
from torch import nn
from d2l import torch as d2l

def train_epoch_my(net, train_iter, loss, updater, lr, parameters, my_model): #my_model判断是否是自己的函数
    """一个迭代周期的训练流程"""
    if isinstance(net, torch.nn.Module):
        net.train() #设置为训练模式
    metric = evaluation.Accumulator(3) #三个累加变量

    for X, y in train_iter:
        if my_model :
            y_hat = net(X, parameters)
        else :
            y_hat = net(X)
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

def train_net_my(net, train_iter, test_iter, loss, num_epochs, updater, parameters=None, lr=0.01, my_model=True): #@save
    """训练模型"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        print("epoch",epoch+1,":")
        train_metrics = train_epoch_my(net, train_iter, loss, updater, lr, parameters, my_model) #返回一个元组
        print(train_metrics)
        test_acc = evaluation.evaluate_accuracy(net, test_iter, parameters, my_model)
        animator.add(epoch + 1, train_metrics + (test_acc,)) #绘图
    train_loss, train_acc = train_metrics
    animator.show_img()

    """当断言不成立时退出"""
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


def init_weights(m, Type='xavier'):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        if Type == 'xavier':
            nn.init.xavier_normal_(m.weight)   #用xavier初始化参数

def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def train_with_GPU(net, train_iter, test_iter, num_epochs, lr, device=None):
    """gpu训练流程"""

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)

    for epoch in range(num_epochs):
        metric = evaluation.Accumulator(3)
        net.train()

        for i, (X,y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l*X.shape[0], evaluation.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0]/metric[2]
            train_acc = metric[1]/metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluation.evaluate_accuracy_GPU(net, test_iter, device)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
            f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
            f'on {str(device)}')

    animator.show_img()
