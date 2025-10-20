from datetime import datetime
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
VALIDATION_SIZE = .2

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

    # X = np.log(X + 1)



    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VALIDATION_SIZE, random_state=parameters.RANDOM_SEED)

    training_loader = DataLoader(BEDDataset(X_train, y_train), batch_size=len(X_train), shuffle=True)
    validation_loader = DataLoader(BEDDataset(X_val, y_val), batch_size=len(X_val), shuffle=False)

    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()
    # loss_fn = torch.nn.PoissonNLLLoss()

    best_val_loss = float("inf")
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
            # print("Sizes:", y.shape, y_hat.shape)
            loss = loss_fn(y_hat.flatten(), y)
            
            # Check for NaN in loss
            
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            # print(y_hat)

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
                # print(y_hat.flatten().numpy())
                # plt.hist([y_hat.flatten().numpy(), y.flatten()] , bins=50, density=True, label=["Predicted", "True"], histtype="bar")
                # plt.legend()
                # plt.show()
                running_vloss += loss.item()
        
        avg_val_loss = running_vloss / (i + 1)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = f"models/model_{timestamp}_BS{BATCH_SIZE}_LR{LEARNING_RATE}_VS{VALIDATION_SIZE}_BIN{parameters.CNN_BIN_SIZE}_W{parameters.SIGNAL_CNN_WINDOW}"
            print(f"Saving as best model in {model_path}")
            torch.save(net.state_dict(), model_path)

            # for name, param in net.named_parameters():
            #     print(name, param)
            
            # exit()

        print('Avg valid loss: {}'.format(avg_val_loss))







