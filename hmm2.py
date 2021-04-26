from localmodel import LocalModel
from learner import Model
import pickle
import random
from copy import deepcopy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

B = 1 # local_minibatch_size
EPOCHS = 2000
PERSONALIZATION_EPOCHS = 1000
lr = 0.01
personalized_lr = 0.001

hidden_dims = 100
num_locs = 28
num_layers = 2
num_users = 20
device = 'cuda' if torch.cuda.is_available()  else 'cpu'



with open(r"old_data.pickle", "rb") as input_file:
    d = pickle.load(input_file)

def get_data(user):
    locs = []
    tims = []
    targs = []
    for data in user:
        locs.append(torch.unsqueeze(data[0], 0))
        tims.append(torch.unsqueeze(data[1], 0))
        targs.append(torch.unsqueeze(data[2], 0))
    tens_locs = torch.cat(locs, dim=0)
    tens_tims = torch.cat(tims, dim=0)
    tens_targs = torch.cat(targs, dim=0)
    return torch.stack((tens_locs, tens_tims), dim=0), tens_targs

train_datas = [get_data(d[u][:60])for u in d]

def get_data(user):
    locs = []
    tims = []
    targs = []
    for data in user:
        locs.append(torch.unsqueeze(data[0], 0))
        tims.append(torch.unsqueeze(data[1], 0))
        targs.append(torch.unsqueeze(data[2], 0))
    tens_locs = torch.cat(locs, dim=0)
    tens_tims = torch.cat(tims, dim=0)
    tens_targs = torch.cat(targs, dim=0)
    return torch.stack((tens_locs, tens_tims), dim=0)[:,:,-1].numpy(), tens_targs.numpy()
test_datas = [get_data(d[u][60:])for u in d]

cat_train = (torch.cat([i[0] for i in train_datas], dim=1)[:,:,-1].numpy(), torch.cat([i[1] for i in train_datas]).numpy())
cat_test = (torch.cat([i[0] for i in test_datas], dim=1)[:,:,-1].numpy(), torch.cat([i[1] for i in test_datas]).numpy())

transition_mat = np.ones((28, 48, 28))

for i in range(len(cat_train[1])):
    samp = cat_train[0][:,i]
    transition_mat[samp[0], samp[1], cat_train[1][i]] +=1
useraccs = []
for u in test_datas:
    corr = 0
    tot = 0
    for j in range(len(u[1])):
        loc = u[0][0,j]
        tim = u[0][1,j]
        targ = u[1][j]
        if targ == np.argmax(transition_mat[loc,tim]):
            corr+=1
        tot+=1
    useraccs.append(corr/tot)