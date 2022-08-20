"""
    Filename : Transformer_Encoder.py
    author   : Yichuan Ma
    date     : 2022/8/14
    reference: dive into deep learning
    description: 实现Transformer_Encoder
"""

import math
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l


# 基于位置的前馈网络(实际上就是一个MLP)
class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


# 整合了残差连接和层规范化的模块
class AddNorm(nn.Module):

    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = dropout
        self.In = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):   # 将输入输出整合
        return self.In(self.dropout(Y)+X)


# 实现编码器
"""
    整体实现的操作为:获取输入，首先进行含残差的自注意力，再进行含残差的多层感知机
    每一层都不会改变输入的形状。
"""
class EncoderBlock(nn.Module):

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = d2l.MultiHeadAttention(key_size, query_size, value_size,
                                                num_hiddens, num_heads, dropout, use_bias)
        # 使用了前面定义的网络
        self.addnorm1 = AddNorm(norm_shape, dropout)
        # 前向的含反馈多层感知机
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens
        )
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


class TransformerEncoder(nn.Module):
    """基于以上模块定义的Transformer编码器"""
    # 多一个num_layers
    def __init__(self,  vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        # 位置编码， 把位置信息直接加进原数据中
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()

        for i in range(num_layers):
            self.blks.add_module(
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                             dropout, use_bias)
            )

    def forward(self, X, valid_lens, *args):
        # 因为位置编码值在-1和1之间，
        # 因此嵌入值乘以嵌入维度的平方根进行缩放，
        # 然后再与位置编码相加。
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)

        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention_weights

        return X
