"""
    Filename : evaluation.py
    author   : Yichuan Ma
    date     : 2022/6/30
    reference: dive into deep learning
"""

import torch
from IPython import display
from d2l import torch as d2l

class Accumulator:  # n个变量的累加器

    def __init__(self, n):
        self.data = [0.0]*n

    def add(self, *args):
        self.data = [a+float(b) for a, b in zip(self.data, args)] #和新的数据累加

    def reset(self):
        self.data = [0.0]*len(self.data)

    def __getitem__(self,idx): #定义数据集的访问方式
        return self.data[idx]

def accuracy(y_hat,y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)  #每行中最大的数作为预测类别
    cmp = y_hat.type(y.dtype) == y #转换数据类型后作比较
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net,data_iter,parameters):
    """评价net在data上的表现"""
    if isinstance(net, torch.nn.Module):  #如果是nn函数
        net.eval()  # 将模型设置为评估模式

    metric = Accumulator(2) #共两个变量：预测数，正确预测数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X, parameters.W, parameters.b), y), y.numel())
    return metric[0]/metric[1]  #返回正确率
