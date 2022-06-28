import torch
import random

"""模型函数实现:loss, forward, batch_size"""

def linreg(X,w,b):  #线性forward
    return torch.matmul(X,w)+b

def squard_loss(y_hat,y): #均方误差,y_hat和y对齐
    return (y_hat-y.reshape(y_hat.shape))**2 /2

def SGD(params,lr,batch_size):   #sgd:随机梯度下降
    with torch.no_grad():  #更新时不参与梯度计算
        for param in params:
            param -= lr*param.grad / batch_size #l求导梯度反传
            param.grad.zero_() #梯度清空

