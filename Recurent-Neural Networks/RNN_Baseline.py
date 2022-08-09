"""
    Filename : RNN_Baseline.py
    author   : Yichuan Ma
    date     : 2022/7/31
    reference: dive into deep learning
    description: Recurrent Neural Networks--Baseline
"""

# RNN的封装和训练

import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import My_training_function.Animator as Animator
import My_training_function.evaluation as evaluation
import RNN_Model as RNN

# 定义预测函数
def predict_RNN(prefix, num_preds, net, vocab, device):
    """在已知量后生成新字符"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))  # 把最后一个输出量作为下一时刻的输入

    for y in prefix[1:]:  # 预热
        _, state = net(get_input(), state)
        outputs.append(vocab[y])

    for _ in range(num_preds):  # 预测
        y, state = net(get_input(), state)
        outputs.append(int((y.argmax(dim=1)).reshape(1)))

    return ''.join([vocab.idx_to_token[i] for i in outputs])  # 转换为字符形式输出


# 梯度裁剪
def grad_clipping(net, theta):
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params

    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))

    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm   # 除以norm2范数


# 训练一个epoch
def train_epoch_RNN(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络"""
    state, timer = None, d2l.Timer()
    metric = evaluation.Accumulator(2)

    for X, y in train_iter:
        if state is None or use_random_iter:
            # 随机初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state是个张量
                state.detach_()
            else:
                # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()

        y = y.T.reshape(-1)
        X, y = X.to(device), y.to(device)

        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()  # 转换为长整型

        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)
        metric.add(l*y.numel(), y.numel())

    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


# train_main_func
def train_RNN(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):

    loss = nn.CrossEntropyLoss()
    animator = Animator.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)

    predict = lambda prefix: predict_RNN(prefix, 50, net, vocab, device)

    # 训练并预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_RNN(net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:   # 输出训练时间
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])

    #animator.show_img()
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))


def test():

    X = torch.arange(10).reshape((2, 5))
    state = net.begin_state(X.shape[0], d2l.try_gpu())
    Y, new_state = net(X.to(d2l.try_gpu()), state)
    print(Y.shape, len(new_state), new_state[0].shape)


def Baseline(type='train'):
    if type == 'train':
        predict_RNN('time traveller ', 10, net, vocab, d2l.try_gpu())
        num_epochs, lr = 500, 1
        train_RNN(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())
    elif type == 'test':
        test()
    else:
        print('输入类型有误。')

if __name__ == '__main__':

    batch_size, num_steps, num_hiddens = 32, 35, 512
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    net = RNN.RNN_Model_My(len(vocab), num_hiddens, d2l.try_gpu(), RNN.get_params,
                           RNN.init_rnn_state, RNN.rnn_forward)
    Baseline()