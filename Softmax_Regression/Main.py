"""
    Filename : Main.py
    author   : Yichuan Ma
    date     : 2022/6/30
    reference: dive into deep learning
    Description: 调用实现的net, evaluation, dataset 综合完成训练流程， 并进行预测。
"""

import torch
import evaluation
from Animator import Animator
import Softmax_Regression
import Dataloader_FashionMNIST
from IPython import display
from d2l import torch as d2l

W = torch.normal(0, 0.01, size=(784, 10), requires_grad=True)
b = torch.zeros(10, requires_grad=True)
parameters = Softmax_Regression.Net_Parameters(W, b)

def train_epoch(net, train_iter, loss, updater, lr):
    """一个迭代周期的训练流程"""
    if isinstance(net, torch.nn.Module):
        net.train() #设置为训练模式
    metric = evaluation.Accumulator(3) #三个累加变量

    for X, y in train_iter:
        y_hat = net(X, parameters.W, parameters.b)
        ls = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer): #使用内置函数
            updater.zero_grad()
            ls.mean().backward()
            updater.step()
        else:
            ls.sum().backward()
            updater(X.shape[0], parameters.W, parameters.b, lr)
    #三个累加量：loss, accuracy, 总个数
    #print(ls)
    metric.add(float(ls.sum()), evaluation.accuracy(y_hat, y), y.numel())
    #print(y.numel())
    #返回训练损失和训练精度
    return metric[0]/metric[2], metric[1]/metric[2]

def train_net(net, train_iter, test_iter, loss, num_epochs, updater, lr=0.01):
    """训练模型"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        print("epoch",epoch+1,":")
        train_metrics = train_epoch(net, train_iter, loss, updater, lr) #返回一个元组
        print(train_metrics)
        test_acc = evaluation.evaluate_accuracy(net, test_iter, parameters)
        animator.add(epoch + 1, train_metrics + (test_acc,)) #绘图
    train_loss, train_acc = train_metrics
    animator.show_img()

    """当断言不成立时退出"""
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

def updater(batch_size, W, b, lr=0.01):
    return Softmax_Regression.sgd([W, b], lr, batch_size)

def main():

    batch_size = 256
    dataset = Dataloader_FashionMNIST.FashionMnist_Dataset(batch_size=batch_size)
    train_iter, test_iter = dataset.data_iter()

    net = Softmax_Regression.net
    loss = Softmax_Regression.cross_entropy
    num_epochs = 5

    train_net(net, train_iter, test_iter, loss, num_epochs, updater, lr=0.1)

if __name__ == "__main__":
    main()