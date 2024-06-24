import sys
from time import time

from torch import IntTensor, Tensor
from torch.nn.modules.loss import _Loss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import Optimizer
from torch.types import Device
from torch.utils.data import DataLoader
from tqdm import tqdm

from task1.evaluate import epoch_accuracy_loss
from task1.model.MyRnn import MyRnn


def train_my_rnn(model: MyRnn, train_dataloader: DataLoader, validation_dataloader: DataLoader,
                 loss_function: _Loss, optimizer: Optimizer, epochs: int,
                 device: Device) -> tuple[MyRnn, list[float], list[float], list[float], list[float]]:
    print("--------------------------------------------------")
    # initialize result holders
    train_accuracies = []
    train_losses = []
    validation_accuracies = []
    validation_losses = []
    # set model to training mode for the duration of the training
    for epoch in range(epochs):
        start = time()
        model.train()
        batch_lengths: IntTensor
        batch_data: Tensor
        batch_train_losses = []
        epoch_train_correct_predictions = 0
        epoch_train_total_predictions = 0
        for batch_lengths, batch_data in tqdm(train_dataloader, unit="item", unit_scale=train_dataloader.batch_size,
                                              file=sys.stdout):
            # zero the gradient each batch
            optimizer.zero_grad()
            # fix data for the model
            batch_data = batch_data.to(device)
            # get the predictions
            outputs: Tensor
            outputs, _ = model((batch_lengths, batch_data), model.create_hidden(train_dataloader.batch_size, device), predict=False)
            # calculate the loss
            packed = pack_padded_sequence(batch_data, batch_lengths, batch_first=True, enforce_sorted=False)
            unpacked, batch_lengths = pad_packed_sequence(packed, batch_first=True)
            loss: Tensor = loss_function(outputs[:, :-2, :].transpose(1, 2), unpacked[:, 1:-1])
            batch_train_losses.append(loss.item())
            # propagate the loss backwards
            loss.backward()
            # update the weights
            optimizer.step()
            # process the predictions
            correct = 0
            count = 0
            for i in range(len(outputs)):
                relevant_outputs = outputs[i, :-2]
                relevant_unpacked = unpacked[i, 1:-1]
                equals = (relevant_outputs.argmax(dim=1) == relevant_unpacked)
                current_correct = equals.sum().item()
                correct += current_correct
                count += equals.size(dim=0)
            # add the correct predictions
            epoch_train_correct_predictions += correct
            # add the total predictions
            epoch_train_total_predictions += count
        # log the epoch time
        epoch_time = time()
        # add the final loss in the epoch to the list
        train_losses.append(sum(batch_train_losses) / len(batch_train_losses))
        # add the final accuracy in the epoch to the list
        train_accuracies.append(epoch_train_correct_predictions / epoch_train_total_predictions)
        # calculate the accuracy and loss on the validation set
        validation_accuracies, validation_losses = epoch_accuracy_loss(model, validation_dataloader, device, loss_function)
        # print the results
        print(f"Epoch: {epoch + 1}")
        print(f"Train Accuracy: {train_accuracies[-1]}, Train Loss: {train_losses[-1]}")
        print(f"Validation Accuracy: {validation_accuracies[-1]}, Validation Loss: {validation_losses[-1]}")
        print(f"Time: {int(epoch_time - start)} (+{int(time() - epoch_time)}) seconds")
        print("--------------------------------------------------")

    return model, train_accuracies, train_losses, validation_accuracies, validation_losses