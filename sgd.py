from localmodel import LocalModel
from learner import Model
import pickle
import random
from copy import deepcopy
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt





# fraction_of_clients_performing_update
# number of training_passes_per_round
B = 1 # local_minibatch_size
EPOCHS = 2000
PERSONALIZATION_EPOCHS = 1000
lr = 0.01
personalized_lr = 0.001

hidden_dims = 100
num_locs = 28
num_layers = 2
num_users = 20

PERFED = True



with open(r"old_data.pickle", "rb") as input_file:
    d = pickle.load(input_file)

global_model = Model(hidden_dims,num_locs,num_layers)

# get data & indx here
# local_models = []
# for u in range(num_users):
#     local_models.append(LocalModel(None, None, B, 3, lr))


# class DatasetSplit(Dataset):
#     """An abstract Dataset class wrapped around Pytorch Dataset class.
#     """
#     def __init__(self, user):
#
#
#         self.dataset = dataset
#
#     def __len__(self):
#         return len(self.idxs)
#
#     def __getitem__(self, item):
#         image, label = self.dataset[self.idxs[item]]
#         return torch.tensor(image), torch.tensor(label)
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

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg
print(PERFED)
all_train_losses = []
all_test_losses = []
all_accs = []
accs = []
for epoch in range(EPOCHS):
    local_weights = []
    local_train_losses = []
    local_test_losses = []
    local_accs = []
    for user in range(num_users):
        user_model = deepcopy(global_model)
        if PERFED:
            w, loss = local_models[user].local_update(user_model, [train_datas[user]], do_hessian=False)
        else:
            w, loss = local_models[user].sgd_update(user_model, [train_datas[user]])
        acc, test_loss = local_models[user].test_model(user_model, [test_datas[user]])

        local_test_losses.append(test_loss)
        local_accs.append(acc)
        local_weights.append(deepcopy(w))
        local_train_losses.append(loss)


    all_train_losses.append(local_train_losses)
    all_accs.append(local_accs)
    all_test_losses.append(local_test_losses)

    global_weights = average_weights(local_weights)

    global_model.load_state_dict(global_weights)
print(PERFED)
torch.save(global_model, "perfed_pre_personalization.pt")

density = 20
means = [np.mean(i) for i in all_train_losses][::density]
stds = [np.std(i) for i in all_train_losses][::density]
x = [i for i, a in enumerate(all_train_losses)][::density]
plt.clf()
plt.errorbar(x, means, stds, linestyle='-')
plt.title("Loss vs. epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig('perfedtrainlosspre.png')

means = [np.mean(i) for i in all_test_losses][::density]
stds = [np.std(i) for i in all_test_losses][::density]
x = [i for i, a in enumerate(all_test_losses)][::density]
plt.clf()
plt.errorbar(x, means, stds, linestyle='-')
plt.title("Loss vs. epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig('perfedtestlosspre.png')

means = [np.mean(i) for i in all_accs][::density]
stds = [np.std(i) for i in all_accs][::density]
x = [i for i, a in enumerate(all_accs)][::density]
plt.clf()
plt.errorbar(x, means, stds, linestyle='-')
plt.title("Validation Accuracy vs. epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.savefig('perfedaccpre.png')
print("fedavg")


plt.clf()
labels = [str(i) for i in range(1,21)]
x = np.arange(len(labels))  # the label locations
width = 0.75  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x, local_accs, width)
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
ax.set_xlabel('User')
ax.set_title('Accuracy by User')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
fig.tight_layout()
plt.savefig("perfeduseraccpre.png")

# global_model = torch.load("fedavg_pre_personalization.pt")

for each in local_models:
    each.lr = personalized_lr

personalized_models = [deepcopy(global_model) for i in range(num_users)]

all_train_losses = []
all_accs = []
all_test_losses = []
for epoch in range(PERSONALIZATION_EPOCHS):
    local_weights = []
    local_train_losses = []
    local_test_losses = []
    local_accs = []
    for user in range(num_users):
        w, loss = local_models[user].sgd_update(personalized_models[user], [train_datas[user]])

        acc, test_loss = local_models[user].test_model(personalized_models[user], [test_datas[user]])

        local_test_losses.append(test_loss)
        local_accs.append(acc)
        local_train_losses.append(loss)

    all_train_losses.append(local_train_losses)
    all_accs.append(local_accs)
    all_test_losses.append(local_test_losses)

density = 20
means = [np.mean(i) for i in all_train_losses][::density]
stds = [np.std(i) for i in all_train_losses][::density]
x = [i for i, a in enumerate(all_train_losses)][::density]
plt.clf()
plt.errorbar(x, means, stds, linestyle='-')
plt.title("Loss vs. epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig('perfedtrainlosspost'+ str(personalized_lr).replace(".","")+'.png')



means = [np.mean(i) for i in all_test_losses][::density]
stds = [np.std(i) for i in all_test_losses][::density]
x = [i for i, a in enumerate(all_test_losses)][::density]
plt.clf()
plt.errorbar(x, means, stds, linestyle='-')
plt.title("Loss vs. epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig('perfedtestlosspost'+ str(personalized_lr).replace(".","")+'.png')

means = [np.mean(i) for i in all_accs][::density]
stds = [np.std(i) for i in all_accs][::density]
x = [i for i, a in enumerate(all_accs)][::density]
plt.clf()
plt.errorbar(x, means, stds, linestyle='-')
plt.title("Validation Accuracy vs. epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.savefig('perfedaccpost'+ str(personalized_lr).replace(".","")+'.png')
print("fedavg")

plt.clf()
labels = [str(i) for i in range(1,21)]
x = np.arange(len(labels))  # the label locations
width = 0.75  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x, local_accs, width)
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
ax.set_xlabel('User')
ax.set_title('Accuracy by User')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
fig.tight_layout()
plt.savefig("perfeduseraccpost"+ str(personalized_lr).replace(".","")+".png")