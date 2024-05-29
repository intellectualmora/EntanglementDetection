# -*- coding: utf-8 -*-
"""
@Time : 2020/7/18 19:47
@Author : mora
@FileName: .py
@Software: PyCharm
@Github ：https://github.com/intellectualmora/
 #  #  #  #  #  #  #  #  #  #  #  #
"""
import torch
import torch.nn as nn

class BPNet(nn.Module):

    def __init__(self, D_in, H, H2, D_out):
        super(BPNet, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.Relu = nn.PReLU()
        self.Tanh = nn.Tanh()
        self.linear2 = nn.Linear(H, H2)
        self.linear3 = nn.Linear(H2, H2)
        self.linear4 = nn.Linear(H2, H)
        self.linear5 = nn.Linear(H, D_out)
        self.loss_fn = nn.MSELoss()
        self.cosloss = nn.CosineEmbeddingLoss()
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=0.0001, weight_decay=1e-5)

    def forward(self, x):
        x = self.linear1(x)
        x = self.Relu(x)
        x = self.linear2(x)
        x = self.Relu(x)
        x = self.linear3(x)
        x = self.Relu(x)
        x = self.linear4(x)
        x = self.Relu(x)
        x = self.linear5(x)
        return x

class RNN_50_100(nn.Module):
    def __init__(self, in_dim, n_y):
        super(RNN_50_100, self).__init__()
        self.hidden_dim = in_dim
        self.Relu = nn.ReLU()
        self.linear1 = nn.Linear(in_dim, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 256)
        self.cell = nn.LSTMCell(input_size=256, hidden_size=256)
        self.linear = nn.Linear(256,n_y)
        self.n_y = n_y
        self.loss_fn = nn.MSELoss()
        self.cos = nn.CosineSimilarity()
        self.optimizer = torch.optim.Adam(self.parameters(),lr=0.005,weight_decay=0.000001)

    def forward(self, x):
        x = self.linear1(x)
        x = self.Relu(x)
        x = self.linear2(x)
        x = self.Relu(x)
        x = self.linear3(x)
        x = self.Relu(x)
        out = None
        h_t = x
        c_t = x
        for i in range(100):
            h_t, c_t = self.cell(h_t, (h_t, c_t))
            bt = torch.reshape(self.linear(h_t),(-1,1,self.n_y))
            if out is None:
                out = bt
            else:
                out = torch.cat((out, bt), dim=1)
        return out