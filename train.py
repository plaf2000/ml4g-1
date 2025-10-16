from model_fc import Net
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import parameters
import numpy as np
import random

random.seed(parameters.RANDOM_SEED)
np.random.seed(parameters.RANDOM_SEED)
torch.manual_seed(parameters.RANDOM_SEED)


BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SIZE = .2

TRAIN_DATA_PATH = "Data/train_data.npz"

class BEDDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


if __name__ == "__main__":
    net = Net()

    training_data = np.load(TRAIN_DATA_PATH)

    X: np.ndarray = training_data["x"]
    y: np.ndarray = training_data["y"]

    X = X.reshape(X.shape[0], -1)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VALIDATION_SIZE, random_state=parameters.RANDOM_SEED)
    training_loader = DataLoader(BEDDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(BEDDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    loss = torch.nn.MSELoss()

    for epoch in range(EPOCHS):

        net.train()
        for X, y in training_loader:
            optimizer.zero_grad()
            outputs = net(X)


