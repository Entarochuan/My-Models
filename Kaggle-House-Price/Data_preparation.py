"""
    Filename : Main_Func.py
    author   : Yichuan Ma
    date     : 2022/7/5
    reference: dive into deep learning
"""

"""
    Main Functions of the codes  
"""


import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
import My_training_function.Download as Download

def Data_preparation():

    DATA_HUB = dict()
    DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

    DATA_HUB['kaggle_house_train'] = (
        DATA_URL + 'kaggle_house_pred_train.csv',
        '585e9cc93e70b39160e7921475f9bcd7d31219ce')

    DATA_HUB['kaggle_house_test'] = (
        DATA_URL + 'kaggle_house_pred_test.csv',
        'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

    train_data = pd.read_csv(Download.download(DATA_HUB, 'kaggle_house_train'))
    test_data = pd.read_csv(Download.download(DATA_HUB, 'kaggle_house_test'))
    return train_data, test_data

def Data_preprocessing(train_data, test_data):

    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:])) #舍弃第一个特征
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index  #处理数值类型的特征
    all_features[numeric_features] = all_features[numeric_features].apply(    #正则化
        lambda x: (x - x.mean()) / (x.std()))

    all_features[numeric_features] = all_features[numeric_features].fillna(0) #补零
    all_features = pd.get_dummies(all_features, dummy_na=True)  #为object类型赋值为独热编码

    num_train = train_data.shape[0]   #在行的方向上重新切分出 train, test
    train_features = torch.tensor(all_features[:num_train].values, dtype=torch.float32)
    test_features = torch.tensor(all_features[num_train:].values, dtype=torch.float32)
    train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

    return train_features, test_features, train_labels

def get_data():    #通过此函数调用data
    train_data, test_data = Data_preparation()
    train_features, test_features, train_labels = Data_preprocessing(train_data, test_data)
    return train_features, test_features, train_labels


