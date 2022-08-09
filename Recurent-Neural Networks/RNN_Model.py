"""
    Filename : RNN_Model.py
    author   : Yichuan Ma
    date     : 2022/7/31
    reference: dive into deep learning
    description: Recurrent Neural Networks--Basic Functions
"""

# 实现了循环神经网络

import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import My_training_function
import My_training_function.evaluation as evaluation
import My_training_function.Animator as Animator

def load_data():
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    return train_iter, vocab


# 模型参数初始化
def get_params(vocab_size, num_hiddens, device):      # 模型的输入和输出长度都是已有词表的长度，输出结果为一个概率分布向量
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层, 可以设置其输出维度
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)  # 数字尔

    # 输出层
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]

    for param in params:
        param.requires_grad_(True)

    return params


# 隐状态初始化
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )


# 定义前向传播函数
def rnn_forward(inputs, state, params):  # inputs里有T个样本

    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []

    # 输入X的形状为:(批量大小， 词表大小)
    # inputs最外的维度代表时间步骤
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh)+b_h)  # 更新隐状态
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)

    return torch.cat(outputs, dim=0), (H,)


# 定义RNN类进行函数封装， 参数生成
class RNN_Model_My:

    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):

        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):  # 调用时完成net传播，也可以自己定义一个forward函数
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)  # 对x每个字符映射为对应vocab的一个独热向量, one_hot是按照字符集编码顺序的。
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)

