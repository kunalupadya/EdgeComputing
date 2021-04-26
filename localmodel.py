#https://github.com/AshwinRJ/Federated-Learning-PyTorch
from typing import List

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from copy import deepcopy


def add_slice(net, step_size = 0.0001, sign = 1):
    for ind, i in enumerate(list(net.parameters())):
        # print(i)
        # print(torch.ones(i.data.shape, requires_grad=True) * step_size*sign)

        i.data = i.data + torch.ones(i.data.shape, requires_grad=True) * .001
        # print(i)
    return net

def gradient_descent(params: List[Tensor],
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
    out_params = []
    for ind, i in enumerate(list(model.parameters())):
        if i.requires_grad:
            first_term.append(i)
    for i, param in enumerate(first_term):
        out_params.append(param.add(second_term[i], alpha=mult))
    return out_params


def replace_weights(model, params):
    j = 0
    for ind, i in enumerate(list(model.parameters())):
        if i.requires_grad:
            i.data = params[j]
            j += 1


class LocalModel():
    def __init__(self, data, idxs, batch_size, num_epochs, lr):
        self.device = 'cuda' if torch.cuda.is_available()  else 'cpu'
        # self.trainloader, self.validloader, self.testloader, = self.train_val_test(data, idxs)
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

        # trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
        #                          batch_size=self.batch_size, shuffle=True)
        # validloader = DataLoader(DatasetSplit(dataset, idxs_val),
        #                          batch_size=int(len(idxs_val)/tvt_split[1]), shuffle=False)
        # testloader = DataLoader(DatasetSplit(dataset, idxs_test),
        #                         batch_size=int(len(idxs_test)/tvt_split[2]), shuffle=False)
        # return trainloader, validloader, testloader
    def sgd_update(self,model, data):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)

        batch_size = len(data[0][1])

        for epoch in range(self.epochs):
            cum_loss = 0
            for batch_idx, (inputs, targets) in enumerate(data):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()

                out = model(inputs)
                loss = self.criterion(out, targets)
                loss.backward()
                # print("loss:" + str(loss))

                optimizer.step()

                cum_loss += loss
        return model.state_dict(), cum_loss.data.item() / (batch_idx + 1)



    def local_update(self, model, data, do_hessian=False):
        model.train()

        # optimizer1 = torch.optim.SGD(model.parameters(), lr=self.lr)
        # optimizer2 = torch.optim.SGD(model.parameters(), lr=self.lr)
        batch_size = len(data[0][1])

        for epoch in range(self.epochs):
            cum_loss = 0
            for batch_idx, (inputs, targets) in enumerate(data):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                b0 = inputs[:,:batch_size//3]
                t0 = targets[:batch_size//3]
                b1 = inputs[:,batch_size//3:(batch_size*2)//3]
                t1 = targets[batch_size//3:(batch_size*2)//3]
                b2 = inputs[:,(batch_size*2)//3:]
                t2 = targets[(batch_size*2)//3:]
                # optimizer1.zero_grad()
                # optimizer2.zero_grad()

                #calculate first gradient descent
                model.zero_grad()
                out = model(b0)
                loss = self.criterion(out, t0)
                loss.backward()
                # print("loss:" + str(loss))
                cum_loss += loss


                params = []
                grads = []
                for ind, i in enumerate(list(model.parameters())):
                    if i.requires_grad:
                        params.append(i)
                        grads.append(i.grad)

                #first gradient descent
                w_tilde = gradient_descent(params, grads, weight_decay=0, lr=self.lr)

                model_tilde = deepcopy(model)
                replace_weights(model_tilde, w_tilde)


                # calculate second gradient descent
                model_tilde.zero_grad()
                out = model_tilde(b1)
                loss = self.criterion(out, t1)
                loss.backward()

                w_tilde_params = []
                w_tilde_grads = []
                for ind, i in enumerate(list(model_tilde.parameters())):
                    if i.requires_grad:
                        w_tilde_params.append(i)
                        w_tilde_grads.append(i.grad)

                #hessian
                if do_hessian:
                    pos_model = deepcopy(model)
                    pos_model.zero_grad()
                    out_params = sum(pos_model, w_tilde_grads, 0.0001)
                    replace_weights(pos_model, out_params)

                    neg_model = deepcopy(model)
                    neg_model.zero_grad()
                    out_params1 = sum(neg_model, w_tilde_grads, -0.0001)
                    replace_weights(neg_model, out_params1)

                    pos_model.zero_grad()
                    out = pos_model(b2)
                    loss = self.criterion(out, t2)
                    loss.backward()

                    pos_model_params = []
                    pos_model_grads = []
                    for ind, i in enumerate(list(pos_model.parameters())):
                        if i.requires_grad:
                            pos_model_params.append(i)
                            pos_model_grads.append(i.grad)

                    neg_model.zero_grad()
                    out1 = neg_model(b2)
                    loss1 = self.criterion(out1, t2)
                    loss1.backward()

                    neg_model_params = []
                    neg_model_grads = []
                    for ind, i in enumerate(list(neg_model.parameters())):
                        if i.requires_grad:
                            neg_model_params.append(i)
                            neg_model_grads.append(i.grad)


                    for i in range(len(pos_model_grads)):
                        pos_model_grads[i] = (pos_model_grads[i] - neg_model_grads[i]) / (2 * 0.0001)
                    hessian = pos_model_grads

                    for i in range(len(w_tilde_grads)):
                        w_tilde_grads[i] = w_tilde_grads[i] - hessian[i]

                new_w = gradient_descent(params, w_tilde_grads, weight_decay=0, lr=self.lr)

                replace_weights(model, new_w)




                # optimizer.step()


        return model.state_dict(), cum_loss.data.item() / (batch_idx+1)

    def test_model(self, model, data):
        model.eval()
        # optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)

        batch_size = len(data[0][1])
        preds = []
        targs = []
        correct = 0
        total = 0
        cum_loss = 0
        for batch_idx, (inputs, targets) in enumerate(data):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            # optimizer.zero_grad()

            out = model(inputs)
            cum_loss += self.criterion(out, targets)
            pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()
            preds.append(pred)
            targs.append(targets)
            total += len(targets)

        test_loss = (cum_loss * batch_size) / total
        acc = 100. * correct / total
        print('Test: Loss: {:.4f}, Accuracy: {:.4f}%'.format( test_loss, acc))
        return acc, test_loss.data.item()