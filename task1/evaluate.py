import sys
from typing import Tuple

import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.types import Device
from torch.utils.data import DataLoader
from tqdm import tqdm

from task1.model.MyRnn import MyRnn


def epoch_accuracy_loss(model: MyRnn, dataloader: DataLoader, device: Device, loss_function: _Loss) -> Tuple[list[float], list[float]]:
    # calculate the accuracy and loss on the validation set
    accuracies = []
    losses = []
    batch_correct_predictions = 0
    batch_total_predictions = 0
    model.eval()
    with torch.no_grad():
        batch_losses = []
        for batch_lengths, batch_data in tqdm(dataloader, unit="item",
                                              unit_scale=dataloader.batch_size, file=sys.stdout):
            # fix data for the model
            batch_data = batch_data.to(device)
            # get the predictions
            packed = pack_padded_sequence(batch_data, batch_lengths, batch_first=True, enforce_sorted=False)
            unpacked, batch_lengths = pad_packed_sequence(packed, batch_first=True)
            outputs: Tensor
            outputs, _ = model((batch_lengths, batch_data),
                               model.create_hidden(dataloader.batch_size, device), predict=False)
            # calculate the loss
            loss: Tensor = loss_function(outputs[:, :-2, :].transpose(1, 2), unpacked[:, 1:-1])
            losses.append(loss.item())
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
            # add the total predictions
            batch_total_predictions += count
            # add the correct predictions
            batch_correct_predictions += correct
            # add the loss to the list
            batch_losses.append(loss.item())
    # get the accuracy of the epoch and put it in the accuracies list
    accuracies.append(batch_correct_predictions / batch_total_predictions)
    # get the loss of the epoch and put it in the losses list
    losses.append(sum(batch_losses) / len(batch_losses))

    return accuracies, losses


def evaluate_model(model: MyRnn, dataloader: DataLoader, device: Device):
    accuracies, losses = epoch_accuracy_loss(model, dataloader, device, torch.nn.CrossEntropyLoss())
    accuracy = torch.tensor(accuracies).mean()
    loss = torch.tensor(losses).mean()
    perplexity = torch.exp(loss)
    return accuracy.item(), loss.item(), perplexity.item()
