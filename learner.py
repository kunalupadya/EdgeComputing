import torch
import torch.nn as nn

num_locs = 28
num_times = 48
num_loc_embeddings =10
num_time_embeddings = 10

class Model(nn.Module):
    def __init__(self, hidden_dim, output_dim, lstm_layers = 2):
        super(Model, self).__init__()

        self.loc_embedding = nn.Embedding(num_locs, num_loc_embeddings)
        self.time_embedding = nn.Embedding(num_times, num_time_embeddings)

        self.lstm = nn.LSTM(num_loc_embeddings+num_time_embeddings, hidden_dim, num_layers=lstm_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # print(x)
        loc = x[0]
        tim = x[1]
        emb1 = self.loc_embedding(loc)
        emb2 = self.time_embedding(tim)

        # print(emb1.shape)

        lstm_in = torch.cat((emb1, emb2), dim=2)
        # print(lstm_in.shape)
        lstm_out, hidden = self.lstm(lstm_in)
        # lstm_out.view(-1, 100).shape

        out = self.fc(lstm_out[:,-1])

        return out

if __name__ == "__main__":
    import pickle
    import random
    import torch.nn as nn
    import torch

    with open(r"old_data.pickle", "rb") as input_file:
        d = pickle.load(input_file)
    n = nn.CrossEntropyLoss()
    m = Model(100, 28)
    x = (torch.unsqueeze(d[0][0][0], 0), torch.unsqueeze(d[0][0][1], 0))
    out = m(x)
    loss = n(out, torch.unsqueeze(d[0][0][2], 0))
    loss.backward()