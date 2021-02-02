import torch.nn as nn


class MultiLayerPerception(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Linear(512, 128),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)
