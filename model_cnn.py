import torch
import torch.nn as nn
import torch.nn.functional as F
import parameters



class GeneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(parameters.N_SIGNALS_CNN, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            # nn.Conv1d(32, 16, kernel_size=3, padding='same'),
            # nn.ReLU(),
            nn.Conv1d(32, 10, kernel_size=5),  # output one number per gene,
            nn.LazyLinear(1),
        )

    def forward(self, x):
        # x shape: (batch, features=2, genes)
        out = self.net(x)
        print(out.shape)
        return out.squeeze(1)  # shape: (batch, genes)


if __name__ == "__main__":
    print(GeneModel())

