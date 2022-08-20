"""
    Filename : Bahdanau Attention.py
    author   : Yichuan Ma
    date     : 2022/8/14
    reference: dive into deep learning
    description: Bahdanau注意力
"""

import torch
from torch import nn
from d2l import torch as d2l

"""
    模型将解码器上一时刻的输出作为查询，将解码器的隐状态作为查询和输入。
"""


class AttentionDecoder(d2l.Decoder):

    def __init__(self, **kwargs):
        super().__init__()

    @property
    def attention_weights(self):
        raise NotImplementedError


class Seq2SeqAttentionDecoder(AttentionDecoder):

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        # self.attention = d2l.AdditiveAttention(
        #     num_hiddens, num_hiddens, num_hiddens, dropout     # 使用之前定义的加性注意力评分函数
        # )
        self.attention = d2l.DotProductAttention(dropout=dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size+num_hiddens, num_hiddens, num_layers, dropout=dropout)   # encoder和本轮输出进行Concat
        self.dense = nn.Linear(num_hiddens, vocab_size)  # 输出层

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        outputs, hidden_state = enc_outputs   # 编码器输出
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # enc_outputs的形状为(batch_size,num_steps,num_hiddens).
        # hidden_state的形状为(num_layers,batch_size,num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        X = self.embedding(X).permute(1, 0, 2)   # 把batch_size放到前面
        outputs, self._attention_weights = [], []

        for x in X:
            # query的形状为(batch_size,1,num_hiddens)
            query = torch.unsqueeze(hidden_state[-1], dim=1)   # 转换形状, 作为查询使用
            # context的形状为(batch_size,1,num_hiddens)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # 在特征维度上连结,作为RNN的输入
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # 将x变形为(1,batch_size,embed_size+num_hiddens)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)

        # 全连接层变换后，outputs的形状为
        # (num_steps,batch_size,vocab_size)
        outputs = self.dense(torch.cat(outputs, dim=0))  # output经过输出层变换
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                          enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights


# 训练
def train():

    # 网络定义
    encoder = d2l.Seq2SeqEncoder(vocab_size=26, embed_size=8, num_hiddens=16,
                                 num_layers=2)
    encoder.eval()
    decoder = Seq2SeqAttentionDecoder(vocab_size=26, embed_size=8, num_hiddens=16,
                                      num_layers=2)
    decoder.eval()
    X = torch.zeros((4, 7), dtype=torch.long)  # (batch_size,num_steps)
    state = decoder.init_state(encoder(X), None)
    output, state = decoder(X, state)
    output.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape

    # 实际训练
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 250, d2l.try_gpu()
    train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
    encoder = d2l.Seq2SeqEncoder(
        len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqAttentionDecoder(
        len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = d2l.EncoderDecoder(encoder, decoder)
    d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)


if __name__ =='__main__':
    train()