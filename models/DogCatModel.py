import torch.nn as nn
from torchvision import models


class Resnet101(nn.Module):

    def __init__(self, n_classes, pretrained=True):
        super(Resnet101, self).__init__()
        self.resnet101 = models.resnet101(pretrained=pretrained)
        self.resnet101.fc = nn.Linear(in_features=2048, out_features=n_classes)

    def forward(self, input_tensor):
        logits = self.resnet101(input_tensor)
        return logits


class Resnet34(nn.Module):

    def __init__(self, n_classes, pretrained=True):

        super(Resnet34, self).__init__()
        self.resnet34 = models.resnet34(pretrained=pretrained)
        self.resnet34.fc = nn.Linear(in_features=512, out_features=n_classes)

    def forward(self, input_tensor):
        logits = self.resnet34(input_tensor)
        return logits
