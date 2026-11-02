"""Small scratch CNN; replace with your baseline or proposed method."""
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, params):
        super().__init__()
        c = params.num_channels
        self.features = nn.Sequential(
            nn.Conv2d(3, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c, 2*c, 3, padding=1), nn.BatchNorm2d(2*c), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(2*c, 4*c, 3, padding=1), nn.BatchNorm2d(4*c), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(4*4*4*c, 4*c), nn.ReLU(),
            nn.Dropout(params.dropout_rate), nn.Linear(4*c, params.num_classes))

    def forward(self, images):
        # [batch, 3, height, width] -> [batch, num_classes] raw logits.
        return self.classifier(self.features(images))


def loss_fn(outputs, labels):
    # CrossEntropyLoss expects raw logits, not softmax probabilities.
    return F.cross_entropy(outputs, labels)


def accuracy(outputs, labels):
    return float(np.mean(np.argmax(outputs, axis=1) == labels))


metrics = {'accuracy': accuracy}
