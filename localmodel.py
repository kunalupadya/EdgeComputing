#https://github.com/AshwinRJ/Federated-Learning-PyTorch
from typing import List

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from copy import deepcopy

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)

def add_slice(net, step_size = 0.0001, sign = 1):
    for ind, i in enumerate(list(net.parameters())):
        # print(i)
        # print(torch.ones(i.data.shape, requires_grad=True) * step_size*sign)

        i.data = i.data + torch.ones(i.data.shape, requires_grad=True) * .001
        print(i)
    return net

def func(params: List[Tensor],
        d_p_list: List[Tensor],
        weight_decay: float,
        lr: float):
    out_params = []
    for i, param in enumerate(params):
        grad = d_p_list[i]
        # w_tilde = param.add(param, alpha=-lr)
        #
        # param.add_(a, alpha=beta)

        d_p = grad
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        out_params.append(param.add(d_p, alpha=-lr))

    return out_params

def sum(model,
        second_term: List[Tensor],
        mult: float):
    first_term = []
    for ind, i in enumerate(list(model.parameters())):
        if i.grad:
            first_term.append(i)
    for i, param in enumerate(first_term):
        param.add_(second_term[i], alpha=mult)


def replace_weights(model, params):
    j = 0
    for ind, i in enumerate(list(model.parameters())):
        if i.grad:
            i.data = params[j]
            j += 1


class LocalModel():
    def __init__(self, data, idxs, batch_size, num_epochs, lr):
        self.device = 'cuda' if torch.cuda.is_available()  else 'cpu'
        self.trainloader, self.validloader, self.testloader, = self.train_val_test(data, idxs)
        self.batch_size = batch_size
        self.epochs = num_epochs
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

    def train_val_test(self, dataset, idxs, tvt_split = (80,10,10)):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(tvt_split[0]*.01*len(idxs))]
        idxs_val = idxs[int(tvt_split[1]*.01*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(tvt_split[2]*.01*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.batch_size, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/tvt_split[1]), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/tvt_split[2]), shuffle=False)
        return trainloader, validloader, testloader

    def local_update(self, model):
        model.train()

        optimizer1 = torch.optim.SGD(model.parameters(), lr=self.lr)
        optimizer2 = torch.optim.SGD(model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            cum_loss = 0
            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                b0 = inputs[:40]
                t0 = targets[:40]
                b1 = inputs[40:80]
                t1 = targets[40:80]
                b2 = inputs[80:]
                t2 = targets[80:]
                # optimizer1.zero_grad()
                # optimizer2.zero_grad()

                #calculate first gradient descent
                out = model(b0)
                loss = self.criterion(out, t0)
                loss.backward()


                params = []
                grads = []
                for ind, i in enumerate(list(model.parameters())):
                    if i.grad:
                        params.append(i)
                        grads.append(i.grad)

                #first gradient descent
                w_tilde = func(params, grads, weight_decay=0, lr=self.lr)

                model_tilde = deepcopy(model)
                replace_weights(model_tilde, w_tilde)



                # calculate second gradient descent
                out = model_tilde(b1)
                loss = self.criterion(out, t1)
                loss.backward()

                w_tilde_params = []
                w_tilde_grads = []
                for ind, i in enumerate(list(model_tilde.parameters())):
                    if i.grad:
                        w_tilde_params.append(i)
                        w_tilde_grads.append(i.grad)

                #second gradient descent
                second_sgd_first_order_params = func(w_tilde_params, w_tilde_grads, weight_decay=0, lr=self.lr) #https://timvieira.github.io/blog/post/2014/02/10/gradient-vector-product/


                #hessian
                pos_model = deepcopy(model)
                sum(pos_model.parameters, w_tilde_grads, 0.0001)
                neg_model = deepcopy(model)
                sum(neg_model.parameters, w_tilde_grads, -0.0001)


                out = pos_model(b2)
                loss = self.criterion(out, t2)
                loss.backward()

                pos_model_params = []
                pos_model_grads = []
                for ind, i in enumerate(list(pos_model.parameters())):
                    if i.grad:
                        pos_model_params.append(i)
                        pos_model_grads.append(i.grad)

                out = neg_model(b2)
                loss = self.criterion(out, t2)
                loss.backward()

                neg_model_params = []
                neg_model_grads = []
                for ind, i in enumerate(list(neg_model.parameters())):
                    if i.grad:
                        neg_model_params.append(i)
                        neg_model_grads.append(i.grad)



                params + delta * w_tilde_grads
                params - delta * w_tilde_grads

                #eval model for both, get grad

                ss = 0.0001
                pos_model = add_slice(deepcopy(model), step_size=ss, sign=1)
                neg_model = add_slice(deepcopy(model), step_size=ss, sign=-1)
                numerator_hessian = pos_model(out) - neg_model(out)
                hessian = numerator_hessian/ss






                # optimizer.step()

                cum_loss += loss

        return model.state_dict(), sum(cum_loss) / len(self.trainloader)


