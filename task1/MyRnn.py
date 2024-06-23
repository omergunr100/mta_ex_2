from typing import Tuple

import torch
from torch import nn, IntTensor, Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.types import Device


class MyRnn(nn.Module):
    def __init__(self, vocab_dim: int, output_size: int, embedding_dim: int, hidden_size: int, n_layers: int,
                 dropout: float):
        super(MyRnn, self).__init__()

        # set self members
        self.vocab_dim = vocab_dim
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        # defining the embedding layer
        self.embedding = nn.Embedding(vocab_dim, embedding_dim, padding_idx=0)
        # defining the lstm layer
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True, num_layers=n_layers,
                            dropout=dropout)
        # defining fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def create_hidden(self, batch_size: int, device: Device):
        weights = next(self.parameters()).data
        hidden = (weights.new(self.n_layers, batch_size, self.hidden_size).zero_().to(device),
                  weights.new(self.n_layers, batch_size, self.hidden_size).zero_().to(device))
        return hidden

    def forward(self, batch: Tuple[IntTensor, Tensor], hidden):
        lengths, data = batch
        # run the batch through the layers
        embeddings = self.embedding(data)
        # pack the padded sequences
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        # run the lstm over the embeddings
        lstm_out, hidden = self.lstm(packed, hidden)
        # unpack the packed sequences
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        # return the output of the lstm
        return self.fc(lstm_out[:, 0:-2, :])

    def predict(self, lstm_out: Tensor, index: int = -1) -> Tensor:
        return self.fc(lstm_out[:, index, :])
