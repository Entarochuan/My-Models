import torch
import random
import matplotlib.pyplot as plt
import linear_regression

class parameters:
    def __init__(self,w,b,lr):
        self.w = w
        self.b = b
        self.lr = lr

class test_dataset:

    def __init__(self, batch_size, labels, features):
        self.batch_size = batch_size
        self.features = features
        self.labels = labels

    def __iter__(self):
        num_data = len(self.features)
        indices = list(range(num_data))
        random.shuffle(indices)
        for i in range(0,num_data,self.batch_size):
            batch_indices = torch.tensor(indices[i:min(i+self.batch_size,num_data)])
            yield self.features[batch_indices], self.labels[batch_indices]

    def data_vision(self):
        plt.figure()
        plt.scatter(self.features[:, (1)].detach().numpy(), self.labels.detach().numpy(),
                        1);  # 加入.detach是为了不保存其梯度信息，以避免网络结构的扩大
        plt.title('data_before')
        plt.show()

    def data_after(self,parameters):
        plt.figure()
        plt.scatter(self.features[:, (1)].detach().numpy(),
                    linear_regression.linreg(self.features,parameters.w,parameters.b).detach().numpy(),
                    1);  # 加入.detach是为了不保存其梯度信息，以避免网络结构的扩大
        plt.title('data_after')
        plt.show()

def synthetic_data(w=torch.tensor([2,-3.4]), b=4.2, num_examples=1000):
    X = torch.normal(0,1,(num_examples,len(w))) #人为生成1000个样本
    y = torch.matmul(X,w)+b
    y += torch.normal(0,0.01,y.shape) #数据集加入高斯噪声干扰
    return X,y.reshape((-1,1))

def train(dataset,lr,num_epochs,p,
          net_type=linear_regression.linreg,
          loss_type=linear_regression.squard_loss):
    net = net_type
    loss = loss_type

    for epoch in range(num_epochs):
        for X,y in dataset:
            l = loss(net(X,p.w,p.b),y)
            l.sum().backward()
            linear_regression.SGD([p.w,p.b],p.lr,dataset.batch_size)
        with torch.no_grad():
            train_l = loss(net(dataset.features, p.w, p.b), dataset.labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')


if __name__ =='__main__':
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)  # 生成数据集X和真实标签y

    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    p = parameters(w,b,lr=0.01)
    dataset = test_dataset(10,labels,features)

    dataset.data_vision()  #查看数据分布情况
    train(dataset,0.01,20,p)
    dataset.data_after(p)   #模型处理特征得到的数据分布情况
