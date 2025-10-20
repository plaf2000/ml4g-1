from datetime import datetime
import os
from model_cnn import GeneModel
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import parameters
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import random

random.seed(parameters.RANDOM_SEED)
np.random.seed(parameters.RANDOM_SEED)
torch.manual_seed(parameters.RANDOM_SEED)


BATCH_SIZE = 2056
EPOCHS = 1000
LEARNING_RATE = 0.001

TRAIN_DATA_PATH = "Data/processed/data_train.npz"

class BEDDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


if __name__ == "__main__":
    net = GeneModel()

    training_data = np.load(TRAIN_DATA_PATH)
    X1_train: np.ndarray = np.load("Data/processed/cnn_input_X1_train.npy")
    X2_train: np.ndarray = np.load("Data/processed/cnn_input_X2_train.npy")

    y_X1_train:  np.ndarray = training_data["labels_X1"]
    y_X2_train: np.ndarray = training_data["labels_X2"]


    X_train = np.concatenate((X1_train, X2_train), axis=0)
    y_train = np.log(np.concatenate((y_X1_train, y_X2_train), axis=0) + 1)


    missing_val = np.any(np.isnan(X_train) | np.isinf(X_train), axis=(1,2))
    X_train = X_train[~missing_val, :, :]
    y_train = y_train[~missing_val]


    X1_val: np.ndarray = np.load("Data/processed/cnn_input_X1_val.npy")
    X2_val: np.ndarray = np.load("Data/processed/cnn_input_X2_val.npy")

    y_X1_val:  np.ndarray = np.load("Data/processed/X1_val_y.npy")
    y_X2_val: np.ndarray = np.load("Data/processed/X2_val_y.npy")


    X_val = np.concatenate((X1_val, X2_val), axis=0)
    y_val = np.log(np.concatenate((y_X1_val, y_X2_val), axis=0) + 1)


    missing_val = np.any(np.isnan(X_train) | np.isinf(X_train), axis=(1,2))
    X_train = X_train[~missing_val, :, :]
    y_train = y_train[~missing_val]



    missing_val = np.any(np.isnan(X_val) | np.isinf(X_val), axis=(1,2))
    X_val = X_val[~missing_val, :, :]
    y_val = y_val[~missing_val]

    training_loader = DataLoader(BEDDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(BEDDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()

    best_val_loss = float("inf")
    model_path = None
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for epoch in range(EPOCHS):
        best_loss = float("inf")
        last_loss = 0

        net.train()

        running_loss = .0
        print(f"Epoch: {epoch + 1}/{EPOCHS}")
        for i, (X, y) in tqdm(enumerate(training_loader), total=len(training_loader)):            
            optimizer.zero_grad()
            y_hat = net(X)
            loss = loss_fn(y_hat.flatten(), y)
            
            loss.backward()

            optimizer.step()
            batch_loss = loss.item()
            running_loss += batch_loss
            best_loss = min(batch_loss, best_loss)
        
        

        avg_loss = running_loss / (i + 1)

        print(f"Training loss: best {best_loss} and avg {avg_loss}")

        net.eval()

        running_vloss = .0

        with torch.no_grad():
            for i, (X, y) in tqdm(enumerate(validation_loader), total=len(validation_loader)):
                y_hat = net(X)
                loss = loss_fn(y_hat.flatten(), y)
                running_vloss += loss.item()
        
        avg_val_loss = running_vloss / (i + 1)
        print('Avg valid loss: {}'.format(avg_val_loss))

        if avg_val_loss < best_val_loss:
            if model_path and os.path.exists(model_path):
                os.remove(model_path)

            model_path = "models/cnn_{}_VLOSS{:.4f}_BS{}_LR{}_BIN{}_W{}".format(timestamp, best_val_loss, BATCH_SIZE, LEARNING_RATE, int(parameters.CNN_BIN_SIZE), int(parameters.SIGNAL_CNN_WINDOW))

            best_val_loss = avg_val_loss
                
            print(f"Saving as best model in {model_path}")
            torch.save(net, model_path)

            # for name, param in net.named_parameters():
            #     print(name, param)
            
            # exit()








