import torch
import torch.nn as nn
import torch.nn.functional as F
import parameters



class GeneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(parameters.N_SIGNALS_CNN, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.LazyConv1d(16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.LazyConv1d(1, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LazyLinear(1),
        )

    def forward(self, x):
        # x shape: (batch, features=2, genes)
        out = self.net(x)
        return out.squeeze(1)  # shape: (batch, genes)


if __name__ == "__main__":
    print(GeneModel())

