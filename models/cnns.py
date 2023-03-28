from torch import nn
from torchvision import models

def get_pretrained_rn_18(num_classes = 1):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, num_classes)
    return model