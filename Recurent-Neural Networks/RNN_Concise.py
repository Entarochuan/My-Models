"""
    Filename : RNN_Concise.py
    author   : Yichuan Ma
    date     : 2022/7/31
    reference: dive into deep learning
    description: Recurrent Neural Networks--Concise Type
"""

# 简洁实现RNN

import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import RNN_Baseline

# 简洁定义RNN网络
class RNNModel(nn.Module):

    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size

        # 讨论是否RNN是双向的
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)

        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens*2, self.vocab_size)


    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)

        output = self.linear((Y.reshape((-1, Y.shape[-1]))))
        return output, state

    def begin_state(self, device, batch_size=1):  # 初始化状态

        # nn.GRU
        if not isinstance(self.rnn, nn.LSTM):
            return torch.zeros((self.num_directions*self.rnn.num_layers,
                                batch_size, self.num_hiddens), device=device)

        # nn.LSTM的隐状态定义,隐状态是一个元组
        return (torch.zeros((
            self.num_directions*self.rnn.num_layers,
            batch_size, self.num_hiddens), device=device),
            torch.zeros(
                (self.num_directions*self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device))


def Baseline():
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    num_hiddens = 256
    rnn_layer = nn.RNN(len(vocab), num_hiddens)
    device = d2l.try_gpu()
    net = RNNModel(rnn_layer, vocab_size=len(vocab))
    net = net.to(device)

    # txt = RNN_Baseline.predict_RNN('time traveller', 10, net, vocab, device)
    # print(txt)

    # train
    num_epochs, lr = 500, 1
    RNN_Baseline.train_RNN(net, train_iter, vocab, lr, num_epochs, device)

if __name__ == '__main__':
    Baseline()


