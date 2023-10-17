import torch
import torch.nn as nn


# Define a custom convolutional block with configurable number of layers
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super(ConvBlock, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# Define the VGG-16 architecture using the ConvBlock class
class VGG16(nn.Module):
    def __init__(self, num_classes=24, n_channels=1):
        super(VGG16, self).__init__()

        self.features = nn.Sequential(
            ConvBlock(n_channels, 64, num_layers=2),
            ConvBlock(64, 128, num_layers=2),
            ConvBlock(128, 256, num_layers=3),
            ConvBlock(256, 512, num_layers=3),
            ConvBlock(512, 512, num_layers=3),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class KeyClassifier(nn.Module):
    def __init__(self, num_classes=24, input_size=200):
        super(KeyClassifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(100, 50),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(50, num_classes)
        )

        self.norm = nn.BatchNorm1d(input_size)

    def forward(self, x):
        x = self.norm(x)
        x = self.classifier(x)
        return x
