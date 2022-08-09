"""
    Filename : Language models & Dataset.py
    author   : Yichuan Ma
    date     : 2022/7/27
    reference: dive into deep learning
    description: 语言模型和数据集
"""

# 在本文件中定义了数据的读取， 数据集生成
# 目标是基于已知数据预测下一个词元

import random
import torch
from d2l import torch as d2l

def seq_data_iter_random(corpus, batch_size, num_steps): # 随机抽样生成子序列
    """使用随机抽样生成一个小批量子序列"""
    corpus = corpus[random.randint(0, num_steps-1):]  # 从随机偏移量出发
    num_subseqs = (len(corpus)-1) // num_steps  # 子序列的个数
    initial_idx = list(range(0, num_subseqs*num_steps, num_steps))  # 每个子序列的起始点
    random.shuffle(initial_idx)  # 随机打乱初始点

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):  # 返回数据迭代器
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_idx[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)


def seq_data_iter_sequential(corpus, batch_size, num_steps):  # 按顺序生成子序列， 以num_step长度前述为数据，移位一时刻后结果作为标签
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y


class SeqDataLoader:  # 整合以上函数，返回字符串数据迭代器
    """加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:   # 指定数据集调用方式， 默认按顺序调用
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_time_machine(batch_size, num_steps,  #@save
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)  # 同时返回迭代器和词表
    return data_iter, data_iter.vocab

