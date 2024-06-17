from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss as Loss


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
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=n_layers, dropout=dropout)
        # defining fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, batch, hidden):
        # run the batch through the layers
        embeddings = self.embedding(batch)
        print(embeddings)
        lstm_out, hidden = self.lstm(embeddings, hidden)
        output = self.fc(lstm_out)
        # return the output and new hidden state
        return output, hidden

    def create_hidden(self, batch_size):
        # create the hidden state
        weight = next(self.parameters()).data
        return (weight.new(self.n_layers, batch_size, self.hidden_size).zero_(),
                weight.new(self.n_layers, batch_size, self.hidden_size).zero_())


def train_my_rnn(model: MyRnn, dataloader: DataLoader, loss_function: Loss, optimizer: Optimizer) -> MyRnn:
    # set model to training mode for the duration of the training
    with model.train():
        for i, batch in enumerate(dataloader):
            # zero the gradient each batch
            optimizer.zero_grad()

    return model
