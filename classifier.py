import torch
import torch.nn as nn

class KeyClassifier(nn.Module):
    def __init__(self, num_classes=24, input_size=200):
        super(KeyClassifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, num_classes),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.classifier(x)
        return x
