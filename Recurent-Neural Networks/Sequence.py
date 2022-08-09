"""
    Filename : Sequence.py
    author   : Yichuan Ma
    date     : 2022/7/20
    reference: dive into deep learning
    description: 建立马尔可夫链序列模型并训练
"""

import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

def Data_Generation():
    T = 1000  # 总共产生1000个点
    time = torch.arange(1, T + 1, dtype=torch.float32)
    x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))  # 生成1000个点的sin加噪图像
    d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
    plt.show()
    return x

def Data_transform(T=1000, x=Data_Generation()):  # 将序列转化为数据-标签格式
    tau = 4
    features = torch.zeros((T-tau, tau))

    for i in range(tau):
        features[:, i] = x[i:T-tau+i]  # 用之前tau位的信息作为输入

    labels = x[tau:].reshape((-1, 1))  # 构造训练集，测试集
    batch_size, n_train = 16, 600
    #前600个数据点用来训练

    train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)
    test_iter  = d2l.load_array((features[n_train:], labels[n_train:]),
                            batch_size, is_train=False)

    return train_iter, test_iter

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)

def get_net():
    net = nn.Sequential(
        nn.Linear(4, 10), nn.ReLU(), nn.Linear(10, 1)
    )
    net.apply(init_weights)

    return net

def train(net, train_iter, loss=nn.MSELoss(reduction='none'), epochs=5, lr=0.01, pred=False):
    trainer = torch.optim.Adam(net.parameters(), lr)

    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()  # 记得清零梯度
            y_hat = net(X)
            l = loss(y_hat, y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch+1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

def Baseline():
    net = get_net()
    train_iter, test_iter = Data_transform()
    train(net, train_iter)


if __name__ == '__main__':
    Baseline()
