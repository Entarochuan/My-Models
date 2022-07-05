"""
    Filename : Date_Synthetic.py
    author   : Yichuan Ma
    date     : 2022/7/5
    reference: dive into deep learning
    Description: 数据生成和迭代器提供
"""


import torch
from torch.utils import data
import random
import matplotlib.pyplot as plt

def synthetic_data(w=torch.tensor([2,-3.4]), b=4.2, num_examples=1000):
    X = torch.normal(0,1,(num_examples,len(w))) #人为生成1000个样本
    y = torch.matmul(X,w)+b
    y += torch.normal(0,0.01,y.shape) #数据集加入高斯噪声干扰
    return X,y.reshape((-1,1))

def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays) #data_arrays输入值为(features,labels)，是一个元组
    return data.DataLoader(dataset, batch_size, shuffle=is_train) #torch.utils库内置的dataLoader函数

