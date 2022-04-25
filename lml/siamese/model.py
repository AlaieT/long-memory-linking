import torch
import torch.nn as nn
from torch.nn import Module
import timm
import torch.nn.functional as F


class SiameseModel(Module):
    def __init__(self) -> None:
        super(SiameseModel, self).__init__()

        self.emb_size = 256

        self.backbone = timm.create_model('efficientnet_b0', pretrained=True)
        in_features = self.backbone.get_classifier().in_features

        fc_name, _ = list(self.backbone.named_modules())[-1]

        if fc_name == 'classifier':
            self.backbone.classifier = nn.Identity()
        elif fc_name == 'head.fc':
            self.backbone.head.fc = nn.Identity()
        elif fc_name == 'fc':
            self.backbone.fc = nn.Identity()
        elif fc_name == 'head.flatten':
            self.backbone.head.fc = nn.Identity()
        elif fc_name == 'head':
            self.backbone.head = nn.Identity()
        else:
            raise Exception('Unknown classifier layer: ' + fc_name)

        self.cosine = nn.CosineSimilarity()
        self.post = nn.Sequential(nn.utils.weight_norm(nn.Linear(in_features, self.emb_size*2), dim=None),
                                  nn.BatchNorm1d(self.emb_size*2),
                                  nn.Dropout(p=0.2),
                                  nn.utils.weight_norm(nn.Linear(self.emb_size*2, self.emb_size)),
                                  nn.BatchNorm1d(self.emb_size))

    def forward(self, first, second):
        out1 = self.post(self.backbone(first))
        out2 = self.post(self.backbone(second))

        return out1, out2
