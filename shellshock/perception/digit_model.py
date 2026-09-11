"""Shared wind-digit architecture for training and lazy runtime loading."""
from torch import nn

class WindDigitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 3, padding=1), nn.ReLU(),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(48 * 12 * 8, 64), nn.ReLU(), nn.Dropout(0.15), nn.Linear(64, 10))

    def forward(self, x):
        return self.classifier(self.features(x))


