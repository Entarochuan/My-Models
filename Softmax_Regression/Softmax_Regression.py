"""
    Filename : Softmax_Regression.py
    author   : Yichuan Ma
    date     : 2022/6/29
    reference: dive into deep learning
"""

"""
In this model, the picture is reshaped into (28*28,1) to be processed by Softmax_Regression
"""

import torch
from IPython import display
from d2l import torch as d2l

class Net_Parameters:

    def __init__(self, W, b, batch_size=256):
        self.W = W
        self.b = b
        self.batch_size = batch_size

def Softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition #应用广播机制，每行转换为一个分布

def net(X,W,b): #X先展平为向量，再Xw+b
    return Softmax(torch.matmul(X.reshape(-1, W.shape[0]), W)+b)


""" 6/30 实现的时候有一个非常愚蠢的bug,忘记取负了，哈哈"""

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])  #数学推导见readme.md

#y记录的是这一行正确的位置

def sgd(params,lr,batch_size):   #sgd:随机梯度下降
    with torch.no_grad():  #更新时不参与梯度计算
        for param in params:
            param -= lr*param.grad / batch_size #l求导梯度反传
            param.grad.zero_() #梯度清空


