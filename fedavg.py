from localmodel import LocalModel
from learner import Model

# fraction_of_clients_performing_update
# number of training_passes_per_round
# local_minibatch_size


# get data & indx here


#for every global epoch
    # iterate through local models and call updates on each for local epoch number of times

    # average weights & update global model, according to the number of samples in the model

net = Model(100, 10)
print(net.parameters())