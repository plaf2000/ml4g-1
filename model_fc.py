import torch
import torch.nn as nn
import torch.nn.functional as F
import parameters



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # nn.BatchNorm1d(parameters.N_FEATURES),  # Batch normalization layer
            nn.Linear(parameters.N_FEATURES, 128),  # First fully connected layer
            nn.Dropout(0.2),        # Dropout layer with 20% dropout rate
            nn.ReLU(),
            nn.LazyLinear(64),       # Second fully connected layer
            nn.Dropout(0.2),        # Dropout layer with 20% dropout rate
            nn.ReLU(),
            nn.LazyLinear(1),        # Output layer
            # nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        # x shape: (batch, features=2, genes)
        out = self.net(x)
        return out  # shape: (batch, genes)
    # def forward(self, x):
    #     x = self.batch_norm(x)  # Apply batch normalization
    #     x = F.relu(self.fc1(x))  # Apply ReLU activation after first layer
    #     x = self.dropout1(x)     # Apply dropout
    #     x = F.relu(self.fc2(x))  # Apply ReLU activation after second layer
    #     x = self.dropout2(x)     # Apply dropout
    #     # x = torch.sigmoid(self.fc3(x))  # Apply sigmoid activation at the output layer
    #     x = F.relu(self.fc3(x))
    #     return x


if __name__ == "__main__":
    print(Net())

