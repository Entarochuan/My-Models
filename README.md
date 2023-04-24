<h1 align="center">
My Models
</h1>
<p align="center">
Self implemented Deep learning models :)
<p align="center">
Reference：《Dive Into Deep Learning》
 <p align="center">
    <a href="https://zh.d2l.ai/">
      <img alt="Tests Passing" src="https://img.shields.io/badge/D2L-%E4%BB%A3%E7%A0%81%E5%8F%82%E8%80%83-brightgreen" />
     <a href="https://zh.d2l.ai/chapter_installation/index.html">
      <img alt="Tests Passing" src="https://img.shields.io/badge/D2L-%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE-brightgreen" />
    </a>


## Introduction
按照[Dive into deep Learning](https://zh.d2l.ai)给出的代码，提供了自己的实现，在一些重要的位置提供了注释。

## Install&Run
如需在本地运行代码，请先按照D2L书给出的环境配置安装所需的包。操作过程中可以将下载源替换为清华源或其它。
     
部分实验结果使用Google Colab运行得到。可以在D2L各章节处直接打开Colab运行。也可以将实现的的`.py`文件修改为`.ipynb`文件格式，上传至Colab并按照下方代码安装所需包后运行。如需上传到Colab，请将My_functions中实现的训练函数替代为d2l包中提供的训练函数。在RNN及以后实现部分中也提供了Colab可直接训练的`.ipynb`文件。

```python
!pip install d2l
!pip install matplotlib==3.0.0
#需要以上的两个包完成绘图和环境导入。
```

