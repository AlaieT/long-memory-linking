import torch
from torch import negative
import torch.nn as nn
from torch.nn import Module
from torchvision.models import mobilenet_v3_small


class SiameseModel(Module):
    def __init__(self) -> None:
        super(SiameseModel, self).__init__()
        self.__backbone = mobilenet_v3_small(pretrained=True)
        self.__to_one = nn.Sequential(nn.Linear(2000, 1), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, first, second):
        out1 = self.__backbone(first)
        out2 = self.__backbone(second)

        out = torch.cat((out1, out2), dim=1)

        out = self.__to_one(out)

        return out
