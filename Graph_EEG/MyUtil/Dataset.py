# -*- coding: utf-8 -*-
"""
Created on Mon May 11 22:06:45 2020

@author: Miz
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch_geometric.nn import SGConv, global_add_pool
from torch_scatter import scatter_add
from torch_geometric.data import Data, Dataset, DataLoader, InMemoryDataset

import numpy as np
import scipy.io as sio

import matplotlib.pyplot as plt

import time

#%% Dataset
        
class SEEDIVdataset(Dataset):
  
    def __init__(self, Dir=''):
        traindata = sio.loadmat(Dir+'SInd_TrainData_8685x62x5.mat')
        self.x = torch.from_numpy(traindata['SInd_TrainData_8685x62x5']).float()
        trainlabel = sio.loadmat(Dir+'SInd_TrainLabel_1x4.mat')
        self.y = torch.from_numpy(trainlabel['SInd_TrainLabel_1x4']).float().reshape(8685,1)
        
        self.yprob = np.zeros((10212,4))
        
        for i,label in enumerate(trainlabel['SInd_TrainLabel_1x4'].reshape(8685,1)) :
            if label[0] == 0 :
                self.yprob[i,:] = [0.85, 0.05, 0.05, 0.05]
            elif label[0] == 1 :
                self.yprob[i,:] = [0.2/3, 1-0.2/3, 0.2/3, 0]
            elif label[0] == 2 :
                self.yprob[i,:] = [0.05, 0.05,0.85, 0.05]
            elif label[0] == 3 :
                self.yprob[i,:] = [0.2/3, 0, 0.2/3, 1-0.2/3]
        
        self.yprob = torch.from_numpy(self.yprob).float()
        
        tmp = sio.loadmat(Dir+'initial_A_62x62.mat')
        self.initialA = torch.from_numpy(tmp['initial_A']).float()
        self.edge_index = torch.from_numpy(tmp['initial_weight_index']).long()

    def __len__(self):
        return np.size(self.y, 0)
 
    def __getitem__(self, idx):
        return Data(x=self.x[idx,:,:],y=self.y[idx,:].reshape(1,self.y.size(1)),yprob=self.yprob[idx,:].reshape(1,self.yprob.size(1)),
                    edge_index=self.edge_index)

class SEEDIVdataset_test(Dataset):
  
    def __init__(self, Dir=''):
        testdata = sio.loadmat(Dir+'SInd_TestData_4080x62x5.mat')
        self.x = torch.from_numpy(testdata['SInd_TestData_4080x62x5']).float()
        testlabel = sio.loadmat(Dir+'SInd_TestLabel_1x4.mat')
        self.y = torch.from_numpy(testlabel['SInd_TestLabel_1x4']).float().reshape(4080,1)
        tmp = sio.loadmat(Dir+'initial_A_62x62.mat')
        self.edge_index = torch.from_numpy(tmp['initial_weight_index']).long()
        
    def __len__(self):
        return np.size(self.y, 0)
 
    def __getitem__(self, idx):
        return Data(x=self.x[idx,:,:],y=self.y[idx,:].reshape(1,self.y.size(1)),edge_index=self.edge_index)