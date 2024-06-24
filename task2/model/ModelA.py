from pathlib import Path

import torch
from torch import nn

from task1.model.MyRnn import MyRnn


class ModelA(nn.Module):
    def __init__(self, pretrained_model: Path):
        super(ModelA, self).__init__()
        # load the pretrained model
        self.pretrained: MyRnn = torch.load(pretrained_model)
        # initialize the classification layers
        self.reduction = nn.Linear(self.pretrained.output_size, self.pretrained.output_size // 2)
        self.classifier = nn.Linear(self.pretrained.output_size // 2, 1)

    def forward(self, x):
        return self.layer(x)

    def predict(self, x):
        return self.layer(x)