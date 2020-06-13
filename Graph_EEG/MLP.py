# -*- coding: utf-8 -*-
"""
Created on Mon May 11 11:14:03 2020

@author: Miz
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch_scatter import scatter_add
from torch.utils.data import Dataset, DataLoader

import numpy as np
import scipy.io as sio

import matplotlib.pyplot as plt

import time

g_batch_size = 16

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fc1 = torch.nn.Linear(62*5,128)
        self.fc2 = torch.nn.Linear(128,256)
        self.fc3 = torch.nn.Linear(256,64)
        self.fc4 = torch.nn.Linear(64,4)
        
    def forward(self,din):
        din = din.x.view(-1,62*5)
        dout = torch.nn.functional.sigmoid(self.fc1(din))
        dout = torch.nn.functional.sigmoid(self.fc2(dout))
        dout = torch.nn.functional.sigmoid(self.fc3(dout))
        return self.fc4(dout)
    
from MyUtil.Dataset import SEEDIVdataset, SEEDIVdataset_test
from MyUtil.Regularization import Regularization

train_dataset = SEEDIVdataset()
train_loader = DataLoader(train_dataset, batch_size=g_batch_size,
                          shuffle=True)

test_dataset = SEEDIVdataset_test()
test_loader = DataLoader(test_dataset, batch_size=300,
                          shuffle=True)

device = torch.device('cuda:0')
mlp = MLP()
mlp.to(device)

optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3 , weight_decay=0)
regu = Regularization(mlp,5e-4,2).to(device)


evalrate=[]
def Eval(record=False) :
    er = 0
    n = 0
    outputall=np.array([],dtype=int)
    labelall=np.array([],dtype=int)
    for i, data in enumerate(test_loader, 0):        
        # get the input
        inputs = data.to(device)
        labels = data.y.to(device)
        mlp.eval()
        
        # forward + backward + optimize
        outputs = mlp(inputs)
        labels = labels# + torch.full( (labels.size(0), labels.size(1)),1e-10 )
        # outputs = F.softmax(outputs,dim=1)
        outputepoch=(outputs.argmax(dim=1))
        labelepoch=(labels.T[0,:].T.long())

        er += (outputepoch==labelepoch).sum().float()
        n += len(labelepoch)
        
        if record :
            outputall=np.append(outputall,outputepoch.cpu().numpy())
            labelall=np.append(labelall,labelepoch.cpu().numpy())
        
    er /= n
    print('EVAL$$$:')
    print(er)
    evalrate.append(er)
    if record:
        return outputall,labelall

lossall=[]
rateall=[]
for epoch in range(800):
    lossepoch = 0
    rateepoch = 0
    n = 0
    starttime = time.time()
    
    if True:#epoch%50 == 0:
        Eval()
    
    for i, data in enumerate(train_loader, 0):   
        mlp.train()
        # get the input
        inputs = data.to(device)
        labels = data.y.to(device)
        labelsprob = data.yprob.to(device)
        
        # RGNN.train()
        # zeros the paramster gradients
        optimizer.zero_grad()       

        # forward + backward + optimize
        outputs = mlp(inputs)
        labels = labels# + torch.full( (labels.size(0), labels.size(1)),1e-10 )
        
        loss = F.kl_div(F.log_softmax(outputs,dim=1), F.softmax(labelsprob,dim=1)) # good one
        # loss = F.kl_div(F.softmax(outputs,dim=1), F.log_softmax(labelsprob,dim=1))
        # loss = kl_categorical(labelsprob, outputs)
        
        # loss = crt2(outputs, labels)
        loss = loss# + regu(mlp)
        
        
        loss.backward()    
        optimizer.step()  
        
        lossall.append( loss.item() )
        
        lossepoch = lossepoch+loss
        rateepoch = rateepoch+((labels.T[0,:].T.long()==outputs.argmax(dim=1)).sum().float()/outputs.size(0))

        n=n+1
    
    # if lossepoch/n <0.035 : 
    #     print("LR low...")
    #     for param_group in optimizer.param_groups:
    #             param_group['lr'] = 2e-5
        
    print("###" + str(epoch))
    print(str(time.time()-starttime) + "sec")
    print(lossepoch/n)
    print(rateepoch/n)
    rateall.append(rateepoch/n)
