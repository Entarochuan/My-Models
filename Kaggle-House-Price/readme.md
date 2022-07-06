# Kaggle-House-Price

[参赛页面](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

在本项目中，根据《Dive Into Deep Learning》提供的baseline优化实现了比赛要求的房价预测项目。

在实际测试中，模型得到了最优为0.14461的表现。

### Innovation

在训练过程中使用了学习率衰减的思路，同时结合`weight_decay`的方法，使模型拥有较好泛化表现的同时更倾向

于走到全局最优的位置。
