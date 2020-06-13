#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 20:19:22 2020

@author: miz
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

from torch.utils.tensorboard import SummaryWriter # to print to tensorboard

#%% Hyper Parameters

g_batch_size = 16

#%% Model Definition

def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes


def add_remaining_self_loops(edge_index,
                             edge_weight=None,
                             fill_value=1,
                             num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index

    mask = row != col
    inv_mask = ~mask
    loop_weight = torch.full(
        (num_nodes, ),
        fill_value,
        dtype=None if edge_weight is None else edge_weight.dtype,
        device=edge_index.device)

    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)
        remaining_edge_weight = edge_weight[inv_mask]
        if remaining_edge_weight.numel() > 0:
            loop_weight[row[inv_mask]] = remaining_edge_weight
        edge_weight = torch.cat([edge_weight[mask], loop_weight], dim=0)

    loop_index = torch.arange(0, num_nodes, dtype=row.dtype, device=row.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index[:, mask], loop_index], dim=1)

    return edge_index, edge_weight


class NewSGConv(SGConv):
    def __init__(self, num_features, num_classes, K=1, cached=False,
                 bias=True):
        super(NewSGConv, self).__init__(num_features, num_classes, K=K, cached=cached, bias=bias)

    # allow negative edge weights
    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        row, col = edge_index
        deg = scatter_add(torch.abs(edge_weight), row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        if not self.cached or self.cached_result is None:
            edge_index, norm = NewSGConv.norm(
                edge_index, x.size(0), edge_weight, dtype=x.dtype)

            for k in range(self.K):
                x = self.propagate(edge_index, x=x, norm=norm)
            self.cached_result = x

        return self.lin(self.cached_result)

    def message(self, x_j, norm):
        # x_j: (batch_size*num_nodes*num_nodes, num_features)
        # norm: (batch_size*num_nodes*num_nodes, )
        return norm.view(-1, 1) * x_j


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class SymSimGCNNet(torch.nn.Module):
    def __init__(self, num_nodes, learn_edge_weight, edge_weight, num_features, num_hiddens, num_classes, K, dropout=0.5, domain_adaptation=""):
        """
            num_nodes: number of nodes in the graph
            learn_edge_weight: if True, the edge_weight is learnable
            edge_weight: initial edge matrix
            num_features: feature dim for each node/channel
            num_hiddens: a tuple of hidden dimensions
            num_classes: number of emotion classes
            K: number of layers
            dropout: dropout rate in final linear layer
            domain_adaptation: RevGrad
        """
        super(SymSimGCNNet, self).__init__()
        self.domain_adaptation = domain_adaptation
        self.num_nodes = num_nodes
        self.xs, self.ys = torch.tril_indices(self.num_nodes, self.num_nodes, offset=0)
        edge_weight = edge_weight.reshape(self.num_nodes, self.num_nodes)[self.xs, self.ys] # strict lower triangular values
        self.edge_weight = nn.Parameter(edge_weight, requires_grad=learn_edge_weight)
        self.dropout = dropout
        self.conv1 = NewSGConv(num_features=num_features, num_classes=num_hiddens[0], K=K)
        self.fc = nn.Linear(num_hiddens[0], num_classes)
        if self.domain_adaptation in ["RevGrad"]:
            self.domain_classifier = nn.Linear(num_hiddens[0], 2)

    def forward(self, data, alpha=0):
        batch_size = len(data.y)
        x, edge_index = data.x, data.edge_index
        edge_weight = torch.zeros((self.num_nodes, self.num_nodes), device=edge_index.device)
        edge_weight[self.xs.to(edge_weight.device), self.ys.to(edge_weight.device)] = self.edge_weight
        edge_weight = edge_weight + edge_weight.transpose(1,0) - torch.diag(edge_weight.diagonal()) # copy values from lower tri to upper tri
        edge_weight = edge_weight.reshape(-1).repeat(batch_size)
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        
        # domain classification
        domain_output = None
        if self.domain_adaptation in ["RevGrad"]:
            reverse_x = ReverseLayerF.apply(x, alpha)
            domain_output = self.domain_classifier(reverse_x)
        x = global_add_pool(x, data.batch, size=batch_size)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # x = F.softmax(self.fc(x), dim=0)
        x = self.fc(x)
        return x, domain_output


from MyUtil.Dataset import SEEDIVdataset, SEEDIVdataset_test
from MyUtil.Regularization import Regularization


writer = SummaryWriter(f'runs//RGNN')

#%% Main

train_dataset = SEEDIVdataset()
train_loader = DataLoader(train_dataset, batch_size=g_batch_size,
                          shuffle=True,pin_memory=True)

test_dataset = SEEDIVdataset_test()
test_loader = DataLoader(test_dataset, batch_size=300,
                          shuffle=True)

RGNN=SymSimGCNNet(62, True, train_dataset.initialA, 5, [128], 4, 3, 0.7)

device = torch.device('cuda:0')
RGNN.to(device)

criterion = nn.KLDivLoss(reduce=True,reduction='mean')
crt1 = nn.CrossEntropyLoss(reduce=True,reduction='sum')
crt2 = nn.MSELoss(reduction='mean')

optimizer = torch.optim.Adam(RGNN.parameters(), lr=1e-4 , weight_decay=0)

regu = Regularization(RGNN,1e-3,2).to(device)

# Test
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
        RGNN.eval()
        
        # forward + backward + optimize
        outputs, domain_output = RGNN(inputs)
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
    else : 
        return er



lossall=[]
rateall=[]

def kl_categorical(p_logit, q_logit):
    p = F.softmax(p_logit, dim=1)
    _kl = torch.sum(p * (F.log_softmax(p_logit, dim=1)
                                   - F.log_softmax(q_logit, dim=1)), 1)
    return torch.mean(_kl)

def main():
    for epoch in range(2000):
        writer.add_histogram('sgc', RGNN.conv1.lin.weight.reshape(-1))
        writer.add_histogram('edge', RGNN.edge_weight)
        writer.add_histogram('fc', RGNN.fc.weight)
        
        lossepoch = 0
        rateepoch = 0
        n = 0
        starttime = time.time()
        
        if True:#epoch%50 == 0:
            evalacc = Eval()
        
        for i, data in enumerate(train_loader, 0):   
            RGNN.train()
            # get the input
            inputs = data.to(device)
            labels = data.y.to(device)
            labelsprob = data.yprob.to(device)
            
            # RGNN.train()
            # zeros the paramster gradients
            optimizer.zero_grad()       
    
            # forward + backward + optimize
            outputs, domain_output = RGNN(inputs)
            labels = labels# + torch.full( (labels.size(0), labels.size(1)),1e-10 )
            
            # loss = criterion(F.log_softmax(labels,dim=0), outputs)
            
            # loss = F.cross_entropy(outputs, labels.T[0,:].T.long())#crt2(outputs, labels) + 1e-7*RGNN.edge_weight.norm(1)
            
            loss = F.kl_div(F.log_softmax(outputs,dim=1), F.softmax(labelsprob,dim=1), reduction='batchmean') # good one
            # loss = F.kl_div(F.softmax(outputs,dim=1), F.log_softmax(labelsprob,dim=1))
            # loss = kl_categorical(labelsprob, outputs)
            
            # loss = crt2(outputs, labels)
            loss = loss + regu(RGNN) + 1e-4*RGNN.edge_weight.norm(1)
            
            
            loss.backward()    
            optimizer.step()  
            
            # print(outputs)
            # print(loss)
            
            
            # print((labels.argmax(dim=1)==outputs.argmax(dim=1)).sum().numpy()/outputs.size(0))
            
            # lossall.append( loss.item() )
            # rateall.append((labels.argmax(dim=1)==outputs.argmax(dim=1)).sum().numpy()/outputs.size(0))
            
            # lossepoch = lossepoch+loss
            # rateepoch = rateepoch+(labels.argmax(dim=1)==outputs.argmax(dim=1)).sum().numpy()/outputs.size(0)
            
            
            lossall.append( loss.item() )
            # rateall.append((labels.T[0,:].T.long()==outputs.argmax(dim=1)).sum().float()/outputs.size(0))
            
            lossepoch = lossepoch+loss
            rateepoch = rateepoch+((labels.T[0,:].T.long()==outputs.argmax(dim=1)).sum().float()/outputs.size(0))
            
            # print(loss)
            # print((labels.T[0,:].T.long()==outputs.argmax(dim=1)).sum().float()/g_batch_size)
            
            n=n+1
        
        # if lossepoch/n <0.035 : 
        #     print("LR low...")
        #     for param_group in optimizer.param_groups:
        #             param_group['lr'] = 2e-5
            
        writer.add_scalar('Training loss', lossepoch/n, global_step = epoch)
        writer.add_scalar('Training Accuracy', rateepoch/n, global_step = epoch)
        writer.add_scalar('Eval Accuracy', evalacc, global_step = epoch)
    
        print("###" + str(epoch))
        print(str(time.time()-starttime) + "sec")
        print(lossepoch/n)
        print(rateepoch/n)
        rateall.append(rateepoch/n)

if __name__=="__main__" :
    main()

def confusion_matrix(preds, labels, conf_matrix):
    # preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix

conf_matrix = torch.zeros(4, 4)


import itertools
# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# test_loader = DataLoader(test_dataset, batch_size=2190,
#                           shuffle=True)

o,l=Eval(True)

# plot_confusion_matrix(confusion_matrix(o,l,conf_matrix), classes=['0','1','2','3'],normalize=True)

