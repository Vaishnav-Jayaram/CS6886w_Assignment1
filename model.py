import torch.nn as nn

ACTIVATIONS = {
    "relu": nn.ReLU(inplace=True),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "silu": nn.SiLU(inplace=True),
    "gelu": nn.GELU(),
}

class VGG6(nn.Module):
    def __init__(self, activation: str = "relu", num_classes: int = 10):
        super().__init__()
        act = ACTIVATIONS[activation]
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), act,
            nn.Conv2d(64, 64, 3, padding=1), act,
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), act,
            nn.Conv2d(128, 128, 3, padding=1), act,
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), act,
            nn.Conv2d(256, 256, 3, padding=1), act,
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))