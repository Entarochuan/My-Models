### Multilayer Perceptrons

在本文件中实现了一个一层结构的多层感知机。

训练流程基于Softmax实现，已封装在`My_training_functions`文件夹下。

模型训练20-epochs, learning rate=0.0035的表现如下图所示:

[![20epoch-0-0035lr.png](https://i.postimg.cc/V6CgPrRK/20epoch-0-0035lr.png)](https://postimg.cc/47XVz3jc)

30-epochs, learning rate=0.0015表现如下图所示:

[![30epoch-0-0015lr.png](https://i.postimg.cc/fLnXXHcv/30epoch-0-0015lr.png)](https://postimg.cc/xk5cV3Fk)

猜测图片在高维流形上loss的形状存在多个局部最小值，学习率较大导致在几个局部最优之间跳跃。
