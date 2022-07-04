### Multilayer Perceptrons

在本文件中实现了一个一层结构的多层感知机。

训练流程基于Softmax实现，已封装在`My_training_functions`文件夹下。

模型训练20-epochs, learning rate=0.0035的表现如下图所示:

[![20epoch-0-0035lr.png](https://i.postimg.cc/V6CgPrRK/20epoch-0-0035lr.png)](https://postimg.cc/47XVz3jc)

30-epochs, learning rate=0.0015表现如下图所示:

[![30epoch-0-0015lr.png](https://i.postimg.cc/fLnXXHcv/30epoch-0-0015lr.png)](https://postimg.cc/xk5cV3Fk)

猜测图片在高维流形上loss的形状存在多个局部最小值，学习率较小导致陷于几个局部最优。

使用learning rate=0.1,训练10-epochs,得到如下所示的结果:

[![10epoch-0-1lr.png](https://i.postimg.cc/x1FjWcqJ/10epoch-0-1lr.png)](https://postimg.cc/NKmwXG6B)

有一定波动，但确实得到了更好的表现。

可以尝试的优化方向: Momentum



#### 简洁实现

在`Concise_Type`内通过torch提供的api简洁搭建了训练baseline，同时对`My_training_functions`作了一定的修改以符合训练的条件。

[![10-0-1.png](https://i.postimg.cc/CLYxjf34/10-0-1.png)](https://postimg.cc/CdP00dLZ)

模型训练表现如图。

