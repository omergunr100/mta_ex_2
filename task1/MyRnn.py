from torch import nn
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
        self.embedding = nn.Embedding(vocab_dim, embedding_dim)
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

    def forward(self, batch, hidden):
        # run the batch through the layers
        embeddings = self.embedding(batch)
        lstm_out, hidden = self.lstm(embeddings, hidden)
        output = self.fc(lstm_out[:, -1, :])
        # return the output and new hidden state
        return output, hidden
