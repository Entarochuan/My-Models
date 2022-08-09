"""
    Filename : Text-Preprocessing.py
    author   : Yichuan Ma
    date     : 2022/7/27
    reference: dive into deep learning
    description: 文本预处理
"""

import collections
import re
from d2l import torch as d2l

d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')


def read_time_machine():
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]   # re.sub用于替换匹配项


def tokenize(lines, token='word'):  # 按照空格划分，获得单词
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)


def count_corpus(tokens):
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):   # 展平列表
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


class Vocab:   # 词表，提供了按字符调用，按频率调用的方法
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens    # '<unk>'表示未知词元
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:  # 按照出现频率加入
            if freq < min_freq:    # 频率小于预设量的词元不纳入考虑
                break
            if token not in self.token_to_idx:   # 查找键值是否存在
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1  # 新建词元

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):   # 调用词表
        # 判断tokens的输入格式
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)    # 以文本为索引查找键值，找不到则返回0
        return [self.__getitem__(token) for token in tokens]  # 返回键值序列

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property   # 修饰为只读属性
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


def load_corpus_time_machine(max_tokens=-1):  # 功能整合
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')   # 按照字符进行词表的划分
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]  # 按照顺序将每个字符投射到里面
    print(corpus)
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


def test(nums=(1, 2, 3, 4)):  # 检查实现

    for num in nums:

        if num == 1:
            lines = read_time_machine()
            print(f'# 文本总行数: {len(lines)}')
            print(lines[0])
            print(lines[10])
            print('check the read part')

        if num == 2:
            tokens = tokenize(lines)
            for i in range(11):
                print(tokens[i])
            print('check the tokenize part')

        if num == 3:
            vocab = Vocab(tokens)
            print(list(vocab.token_to_idx.items())[:10])
            print('check the vocab part')

        if num == 4:
            for i in [0, 10]:
                print('文本:', tokens[i])
                print('索引:', vocab[tokens[i]])
            corpus, vocab = load_corpus_time_machine()
            print(len(corpus), len(vocab))
            print('Final Check done')
    # corpus, vocab = load_corpus_time_machine()

if __name__ == '__main__':
    test()
