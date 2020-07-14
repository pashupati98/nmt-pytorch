import torch
import torch.nn as nn
#from embedding import *
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, layers=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.layers = layers
        # -----------------------------------------------------------------
        # self.weight_matrix = create_weight_matrix()
        # self.embedding, num_embeddings, embedding_dim = create_emb_layer(self.weight_matrix, True)
        self.embedding = nn.Embedding(input_size, hidden_size, layers)

        # -------------------------------------------------------------------
        # self.gru = nn.GRU(embedding_dim, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, layers)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.layers, 1, self.hidden_size, device=device), torch.zeros(self.layers, 1, self.hidden_size, device=device)

# end of file
