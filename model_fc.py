import torch
import torch.nn as nn
import torch.nn.functional as F
import parameters



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.batch_norm = nn.BatchNorm1d(parameters.N_FEATURES)  # Batch normalization layer
        self.fc1 = nn.Linear(parameters.N_FEATURES, 256)  # First fully connected layer
        self.dropout1 = nn.Dropout(0.2)        # Dropout layer with 20% dropout rate
        self.fc2 = nn.Linear(256, 128)       # Second fully connected layer
        self.dropout2 = nn.Dropout(0.2)        # Dropout layer with 20% dropout rate
        self.fc3 = nn.Linear(128, 1)        # Output layer
    def forward(self, x):
        x = self.batch_norm(x)  # Apply batch normalization
        x = F.relu(self.fc1(x))  # Apply ReLU activation after first layer
        x = self.dropout1(x)     # Apply dropout
        x = F.relu(self.fc2(x))  # Apply ReLU activation after second layer
        x = self.dropout2(x)     # Apply dropout
        # x = torch.sigmoid(self.fc3(x))  # Apply sigmoid activation at the output layer
        x = F.relu(self.fc3(x))
        return x


if __name__ == "__main__":
    print(Net())

