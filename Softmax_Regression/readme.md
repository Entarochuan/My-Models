# Softmax_Regression

Reference : 《Dive Into Deep Learning》

基于Fashion_Mnist数据集训练了一个Softmax回归模型。

下面是各个`.py`文件的功能:

| 文件名                | 实现功能                                       |
| --------------------- | ---------------------------------------------- |
| Animator.py           | D2L定义的绘图工具，记录变化量并绘制折线图      |
| Dataloader_...  .py   | 下载并图像数据库并提供数据集的迭代器           |
| evaluation.py         | 实现了对变量统计的累加器，计算结果准确率的函数 |
| Softmax_Regression.py | 实现Softmax回归，使用交叉熵损失函数            |
| Main.py               | 实现模型初始化及训练，整合以上模块             |



### 模型表现

[![Performance.png](https://i.postimg.cc/4xbD51cc/Performance.png)](https://postimg.cc/Xrq1NdQN)

learning rate = 0.1, 训练5个epoch后，在测试集上的准确率达到 85.42 %

### 

### 交叉熵实现

```python
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])  #数学推导见readme.md
```

这里y为一个向量，记录的是当前行内正确标签的位置(0-9)。


