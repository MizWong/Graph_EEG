import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch_geometric.nn import SGConv, global_add_pool, ChebConv, GraphConv, BatchNorm, InstanceNorm
from torch_scatter import scatter_add
from torch_geometric.data import Data, Dataset, DataLoader, InMemoryDataset

import numpy as np
import scipy.io as sio

import matplotlib.pyplot as plt

import time

from torch.utils.tensorboard import SummaryWriter # to print to tensorboard

#%% Hyper Parameters

# batch_size = 32

class EEGNet(torch.nn.Module):
    def __init__(self, num_nodes, learn_edge_weight, edge_weight, num_features, num_hiddens, num_classes, K, dropout=0.5):
        super(EEGNet, self).__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_hiddens = num_hiddens
        self.xs, self.ys = torch.tril_indices(self.num_nodes, self.num_nodes, offset=0)
        edge_weight = edge_weight.reshape(self.num_nodes, self.num_nodes)[self.xs, self.ys] # strict lower triangular values
        # edge_weight_gconv = torch.zeros(self.num_hiddens,1)
        # self.edge_weight_gconv = nn.Parameter(edge_weight_gconv, requires_grad=True)
        # nn.init.xavier_uniform_(self.edge_weight_gconv)
        self.edge_weight = nn.Parameter(edge_weight, requires_grad=learn_edge_weight)
        self.dropout = dropout
        self.chebconv_single = ChebConv(num_features, 1, K, node_dim=0)
        self.chebconv0 = ChebConv(num_features, num_hiddens[0], K, node_dim=0)
        self.chebconv1 = ChebConv(num_hiddens[0], 1, K, node_dim=0)

        # self.fc1 = nn.Linear(num_nodes, num_hiddens)
        self.fc2 = nn.Linear(num_nodes, num_classes)

    def forward(self, data) :
        batch_size = len(data.y)
        x, edge_index = data.x, data.edge_index
        # edge_weight_gconv = self.edge_weight_gconv.reshape(-1).repeat(batch_size)
        edge_weight = torch.zeros((self.num_nodes, self.num_nodes), device=edge_index.device)
        edge_weight[self.xs.to(edge_weight.device), self.ys.to(edge_weight.device)] = self.edge_weight
        edge_weight = edge_weight + edge_weight.transpose(1,0) - torch.diag(edge_weight.diagonal()) # copy values from lower tri to upper tri
        edge_weight = edge_weight.reshape(-1).repeat(batch_size)
        # x = F.relu(self.chebconv0(x, edge_index, edge_weight, data.batch))
        # x = self.chebconv1(x, edge_index, edge_weight, data.batch)
        x = self.chebconv0(x, edge_index, edge_weight, data.batch)
        x = self.chebconv1(x, edge_index, edge_weight, data.batch)

        # x = torch.matmul(x, self.edge_weight_gconv)
        # x = F.relu(x.view(batch_size, self.num_nodes*self.num_features))
        x = x.view(batch_size, self.num_nodes)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        
        return x
        

from MyUtil.Dataset import SEEDIVdataset, SEEDIVdataset_test
from MyUtil.Regularization import Regularization

device = torch.device('cuda:0')


def Eval(record=False) :
    er = 0
    n = 0
    outputall=np.array([],dtype=int)
    labelall=np.array([],dtype=int)
    for i, data in enumerate(test_loader, 0):        
        # get the input
        inputs = data.to(device)
        labels = data.y.to(device)
        EEG.eval()
        
        # forward + backward + optimize
        outputs = EEG(inputs)
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

batch_sizes = [32]
learning_rates = [1e-5]
epochs = [300]

for batch_size in batch_sizes:
    for lrtestcase,learning_rate in enumerate(learning_rates,0):
        # Initialization
        writer = SummaryWriter(f'runs/ChebNetEEG-Batch{batch_size}-LR{learning_rate}')

        train_dataset = SEEDIVdataset()
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=True,pin_memory=True)

        test_dataset = SEEDIVdataset_test()
        test_loader = DataLoader(test_dataset, batch_size=300,
                                shuffle=True)

        EEG = EEGNet(62, True, train_dataset.initialA, 5, [64,64,1], 4, 5, 0.7)

        EEG.to(device)

        optimizer = torch.optim.Adam(EEG.parameters(), lr=learning_rate , weight_decay=0)

        regu = Regularization(EEG,1e-3,2).to(device)

        evalrate=[]
        lossall=[]
        rateall=[]

        for epoch in range(epochs[lrtestcase]):
            
            # writer.add_histogram('gconv', EEG.edge_weight_gconv)
            # writer.add_histogram('cheb0', EEG.chebconv0.weight.reshape(-1))
            # writer.add_histogram('cheb1', EEG.chebconv1.weight.reshape(-1))
            # writer.add_histogram('cheb2', EEG.chebconv2.weight.reshape(-1))
            writer.add_histogram('cheb', EEG.chebconv_single.weight.reshape(-1))
            writer.add_histogram('edge', EEG.edge_weight)
            writer.add_histogram('fc', EEG.fc2.weight)

            lossepoch = 0
            rateepoch = 0
            n = 0
            starttime = time.time()
            
            evalacc = Eval()
            
            for i, data in enumerate(train_loader, 0):   
                EEG.train()
                # get the input
                inputs = data.to(device)
                labels = data.y.to(device)
                labelsprob = data.yprob.to(device)
                
                # RGNN.train()
                # zeros the paramster gradients
                optimizer.zero_grad()       

                # forward + backward + optimize
                outputs = EEG(inputs)
                labels = labels# + torch.full( (labels.size(0), labels.size(1)),1e-10 )

                loss = F.kl_div(F.log_softmax(outputs,dim=1), F.softmax(labelsprob,dim=1),reduction='batchmean') \
                    + 0#regu(EEG)# good one
                
                loss.backward()    
                optimizer.step()  
                
                lossall.append( loss.item() )

                lossepoch = lossepoch+loss
                rateepoch = rateepoch+((labels.T[0,:].T.long()==outputs.argmax(dim=1)).sum().float()/outputs.size(0))

                n=n+1
            
            writer.add_scalar('Training loss', lossepoch/n, global_step = epoch)
            writer.add_scalar('Training Accuracy', rateepoch/n, global_step = epoch)
            writer.add_scalar('Eval Accuracy', evalacc, global_step = epoch)

            print("###" + str(epoch))
            print(str(time.time()-starttime) + "sec")
            print(lossepoch/n)
            print(rateepoch/n)
            rateall.append(rateepoch/n)

        writer.add_hparams({'lr': learning_rate, 'bsize': batch_size}, 
                   {'trainacc': sum(rateall)/len(rateall),
                    'evalacc': sum(evalrate)/len(evalrate),
                    'loss': sum(rateall)/len(rateall),
                    'maxeval': max(evalrate)
                    })


            