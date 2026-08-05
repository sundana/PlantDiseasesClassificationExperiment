import torch.nn as nn
from torchvision import models

def get_mobilenet_v2(num_classes=1000):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model

def get_shufflenet_v2(num_classes=1000):
    model = models.shufflenet_v2_x1_0(weights=None)
    model.fc = nn.Linear(1024, num_classes)
    return model

def get_efficientnet_b0(num_classes=1000):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
