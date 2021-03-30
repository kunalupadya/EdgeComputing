import torch
import torch.nn as nn

num_locs = 10
num_times = 24
num_loc_embeddings =5
num_time_embeddings = 8

class Model(nn.Module):
    def __init__(self, hidden_dim, output_dim, lstm_layers = 2):
        super(Model, self).__init__()

        self.loc_embedding = nn.Embedding(num_locs, num_loc_embeddings)
        self.time_embedding = nn.Embedding(num_times, num_time_embeddings)

        self.lstm = nn.LSTM(num_loc_embeddings+num_time_embeddings, hidden_dim, num_layers=lstm_layers)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        emb1 = self.loc_embedding(x[0])
        emb2 = self.time_embedding(x[1])

        lstm_in = torch.cat(emb1, emb2)
        lstm_out = self.lstm(lstm_in)

        out = self.fc(lstm_out)

        return out

