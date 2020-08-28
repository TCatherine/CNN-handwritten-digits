import torch.nn as nn
import torch
import numpy as np
from Convolution import сonvolution

class Net(nn.Module):
    # @property
    def __init__(self):
        super(Net, self).__init__()
        self.con1=nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.fun1=nn.ReLU()
        self.pool1=nn.MaxPool2d(kernel_size=4, stride=4)
        self.con2=nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=3, padding=1)
        self.fun2=nn.ReLU()
        self.pool2=nn.MaxPool2d(kernel_size=2, stride=2)
        self.fully_conected=nn.Sequential(nn.Linear(256, 1024),nn.ReLU(),nn.Linear(1024, 10))
        self.softmax=nn.Softmax(1)

    def forward(self, x):
        con1=self.con1(x)
        fun1=self.fun1(con1)
        #res1=self.pool1(fun1)
        con2=self.con2(fun1)
        fun2=self.fun2(con2)
        res2=self.pool2(fun2)
        var=res2.reshape(res2.shape[0], -1)
        variable = self.fully_conected(var)
        out_sm = self.softmax(variable)
        return variable, out_sm

class Model:
    def __init__(self, name):
        self.name = name
        self.net = Net()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.1)
        #self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.1)
        self.loss_func = torch.nn.CrossEntropyLoss()

    def Loss(self, x, y_true):
        x = torch.FloatTensor(np.array(x))
        y_true = torch.FloatTensor(np.array(y_true))
        y_pred, y_pred_sm = self.net(x)
        #y= сonvolution(x)
        y_pred = torch.FloatTensor(y_pred)
        loss = self.loss_func(y_pred, torch.max(y_true, 1)[1])
        return loss

    def Train(self, train_x, train_y):
        L = self.Loss(train_x, train_y)
        self.optimizer.zero_grad()
        L.backward()
        self.optimizer.step()

