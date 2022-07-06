"""
    Filename : Dataloader_FashionMNIST.py
    author   : Yichuan Ma
    date     : 2022/6/29
    reference: dive into deep learning
"""

import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

class FashionMnist_Dataset:
    # flag标明是否需要数据集下载
    def __init__(self, flag=True, trans=transforms.ToTensor(),
                 batch_size=256, workers=0):
        print('creating dataset...')
        self.flag = flag
        # 将导入的图像转换为32位浮点，除以255(归一化)
        self.trans = trans
        self.batch_size = batch_size
        self.workers = workers

        if self.flag :
            self.train = torchvision.datasets.FashionMNIST(
                root='./data', train=True, transform=self.trans, download=True)
            self.test = torchvision.datasets.FashionMNIST(
                root='./data', train=False, transform=self.trans, download=True)
            self.flag = False
        print('dataset prepared!')
    def get_dataLoader_workers(self):
        return self.workers

    def data_iter(self, resize=None): #resize参数支持将图像调整大小
        if resize :
            trans = [self.trans].insert(0, transforms.Resize(resize))
            self.trans = transforms.Compose(trans)
        return(data.DataLoader(self.train, self.batch_size, shuffle=True,
                               num_workers=self.get_dataLoader_workers(), ),
               data.DataLoader(self.train, self.batch_size, shuffle=False,
                               num_workers=self.get_dataLoader_workers(), ),
              )

#test if the dataset is correct
def test():
    dataset = FashionMnist_Dataset()
    train_iter, test_iter = dataset.data_iter()
    timer = d2l.Timer() #调用了d2l包中自带的计时器
    for X, y in train_iter:
        continue
    print(f'{timer.stop():.2f} sec')

if __name__ == "__main__":
    test()