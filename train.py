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


EPOCHS = 100
LEARNING_RATE = 0.001

TRAIN_DATA_PATH = "Data/train_data.npz"

class BEDDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


if __name__ == "__main__":
    net = Net()

    training_data = np.load(TRAIN_DATA_PATH)

    X = training_data["x"]
    y = training_data["y"]  

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=parameters.RANDOM_SEED)
    training_loader = DataLoader(BEDDataset(X_train, y_train), batch_size=32, shuffle=True)
    validation_loader = DataLoader(BEDDataset(X_val, y_val), batch_size=32, shuffle=False)


    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    loss = torch.nn.MSELoss()

    for epoch in range(EPOCHS):

        net.train()
        for data, labels in training_loader:
            optimizer.zero_grad()
            outputs = net(data)


