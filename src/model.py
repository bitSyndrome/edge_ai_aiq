import torch
import torch.nn as nn

MODEL_REGISTRY = {}


def register_model(cls):
    MODEL_REGISTRY[cls.__name__] = cls
    return cls


@register_model
class AirQualityCNN(nn.Module):
    """공기질 분류를 위한 경량 1D-CNN.

    Input:  (batch, window_size, num_features)  e.g. (B, 30, 5)
    Output: (batch, num_classes)                e.g. (B, 4)
    """

    def __init__(self, num_features=5, window_size=30, num_classes=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(num_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        # x: (batch, window, features) -> (batch, features, window) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv(x).squeeze(-1)  # (batch, 64)
        return self.classifier(x)


@register_model
class AirQualityMLP(nn.Module):
    """공기질 분류를 위한 경량 MLP.

    Input:  (batch, window_size, num_features)  e.g. (B, 30, 5)
    Output: (batch, num_classes)                e.g. (B, 4)
    """

    def __init__(self, num_features=5, window_size=30, num_classes=4):
        super().__init__()
        input_dim = window_size * num_features  # 30 * 5 = 150
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (batch, window, features) -> (batch, window * features)
        x = x.reshape(x.size(0), -1)
        return self.net(x)
