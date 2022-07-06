"""
    Filename : Train.py
    author   : Yichuan Ma
    date     : 2022/7/5
    reference: dive into deep learning
    description: 搭建baseline, 按照match要求定义损失函数
"""

import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
import My_training_function.Download as Download
import Data_preparation as Data

def log_rmse(net, features, labels, loss=nn.MSELoss()):
    clipped_preds = torch.clamp(net(features), 1, float('inf'))  #稳定取值
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size, loss=nn.MSELoss()):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        if (epoch+1) % 20 == 0:   # 每20次lr减少一次
            learning_rate = learning_rate * 0.92
            optimizer = torch.optim.Adam(net.parameters(),
                                         lr=learning_rate, weight_decay=weight_decay)
            if (epoch+1) % 100 == 0:
                learning_rate = learning_rate + 0.3
                print(f'lr changed into{float(learning_rate)}')
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))

        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))

    return train_ls, test_ls

def get_net(in_features):
    net = nn.Linear(in_features, 1)
    return net

def get_k_fold_data(k, i, X, y):

    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j*fold_size, (j+1)*fold_size) #切片:起始，结束
        X_part, y_part = X[idx, :], y[idx]
        if j == i :
            X_valid, y_valid = X_part, y_part             #测试集
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.concat([X_train, X_part], 0)
            y_train = torch.concat([y_train, y_part], 0)  #拼接得到训练集

    return X_train, y_train, X_valid, y_valid

def k_fold_train(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size, in_features, loss=nn.MSELoss()):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        print(f'{i + 1}折训练中...')
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net(in_features)
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        print(f'{i + 1}折训练完毕')
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k

def train_baseline():
    train_features, test_features, train_labels = Data.get_data()

    #loss = nn.MSELoss()
    loss = nn.CrossEntropyLoss()
    in_features = train_features.shape[1]  #总共纳入考虑的列数(即特征个数)
    #net = get_net(in_features)
    # #net = nn.Sequential(nn.Linear(in_features, 1))
    k, num_epochs, lr, weight_decay, batch_size = 5, 1000, 15, 0.06, 128
    train_l, valid_l = k_fold_train(k, train_features, train_labels, num_epochs, lr,
                              weight_decay, batch_size, in_features, loss)
    print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
          f'平均验证log rmse: {float(valid_l):f}')

def train_and_pred(in_features, train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net(in_features)
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    # 将网络应用于测试集。
    preds = net(test_features).detach().numpy()
    # 将其重新格式化以导出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)

if __name__ =='__main__':
    train_data, test_data = Data.Data_preparation()
    num_epochs, lr, weight_decay, batch_size = 4000, 100, 0.055, 128
    train_features, test_features, train_labels = Data.get_data()
    in_features = train_features.shape[1]
    train_and_pred(in_features, train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size)