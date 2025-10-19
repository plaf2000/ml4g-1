import torch
import torch.nn as nn
import torch.nn.functional as F
import parameters



class GeneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(7, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=1)  # output one number per gene
        )

    def forward(self, x):
        # x shape: (batch, features=2, genes)
        out = self.net(x)
        return out.squeeze(1)  # shape: (batch, genes)


if __name__ == "__main__":
    print(GeneModel())

