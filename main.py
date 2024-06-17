from torch import optim, nn
from torch.utils.data import DataLoader

from task1.IMDBDataset import IMDBDataset
from task1.MyRnn import MyRnn
from task1.process_data import process_data, create_vocabulary
from nltk.tokenize import word_tokenize


if __name__ == '__main__':
    # process the data into dataframes
    df_train, df_test = process_data(data_dir="data/original/aclImdb", save_dir="data/processed",
                                     tokenizer=word_tokenize)
    # create the vocabulary object
    vocabulary = create_vocabulary(df_train)
    # create the dataset and loader
    dataset = IMDBDataset(df=df_train)
    dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)
    # create instance of my rnn
    vocab_size = len(vocabulary)
    model = MyRnn(vocab_dim=vocab_size, output_size=vocab_size, embedding_dim=50, n_layers=2, dropout=0.5,
                  hidden_size=100)
    # train the model
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
