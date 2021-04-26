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
test_datas = [get_data(d[u][60:])for u in d]

cat_train = (torch.cat([i[0] for i in train_datas], dim=1), torch.cat([i[1] for i in train_datas]))
cat_test = (torch.cat([i[0] for i in test_datas], dim=1), torch.cat([i[1] for i in test_datas]))

global_model = Model(hidden_dims,num_locs,num_layers)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    local_weights = []
    local_train_losses = []
    local_test_losses = []
    local_accs = []

    model = global_model
    data = [cat_test]
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    batch_size = len(data[0][1])

    for epoch in range(3):
        cum_loss = 0
        for batch_idx, (inputs, targets) in enumerate(data):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            out = model(inputs)
            loss = criterion(out, targets)
            loss.backward()
            # print("loss:" + str(loss))

            optimizer.step()

            cum_loss += loss
    local_train_losses.append(cum_loss.data.item() / (batch_idx + 1))

    for user in range(num_users):
        data = [test_datas[user]]
        model = global_model
        model.eval()

        batch_size = len(data[0][1])
        preds = []
        targs = []
        correct = 0
        total = 0
        cum_loss = 0
        for batch_idx, (inputs, targets) in enumerate(data):
            inputs, targets = inputs.to(device), targets.to(device)

            out = model(inputs)
            cum_loss += criterion(out, targets)
            pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()
            preds.append(pred)
            targs.append(targets)
            total += len(targets)

        test_loss = (cum_loss * batch_size) / total
        acc = 100. * correct / total
        print('Test: Loss: {:.4f}, Accuracy: {:.4f}%'.format( test_loss, acc))
        local_accs.append(acc)
        local_test_losses.append(test_loss.data.item())