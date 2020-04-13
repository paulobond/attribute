from torchvision import models
import torch
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import pickle
from torch.utils.data import Dataset
from collections import defaultdict
from torch.optim import lr_scheduler
from attribute_index import attribute2attribute_index, n_attributes, attribute_index2attribute
from attribute_index import n_attributes


use_cuda = torch.cuda.is_available()

class Model(nn.Module):
    """
    Using resnet50 + freeze the first layers
    """

    def __init__(self):

        super(Model, self).__init__()

        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, n_attributes)

        # Freezing the first layers of the network
        freezed_layers = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1']
        for layer_name, module in self.model.named_children():
            if layer_name in freezed_layers:
                for param in module.parameters():
                    param.require_grad = False
            else:
                for param in module.parameters():
                    param.require_grad = True

    def unfreeze(self):
        for layer_name, module in self.model.named_children():
            for param in module.parameters():
                param.require_grad = True

    def forward(self, x):
        return torch.sigmoid(self.model(x))


