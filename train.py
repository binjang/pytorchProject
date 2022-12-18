import os
import time

import numpy as np
import pandas as pd
import yfinance as yf

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from tqdm import tqdm

device = torch.device('mps' if torch.has_mps else 'cpu')

ticker = "NVDA"

dataset = yf.download(ticker)

NVDA['ma7'] = NVDA['Close'].rolling(window=5).mean()
NVDA['ma30'] = NVDA['Close'].rolling(window=20).mean()

X = NVDA.loc[:, NVDA.columns != "Close"]
Y = NVDA.loc[:, "Close"]

X = X[20:]
Y = Y[20:]

X_train, X_test = X[: len(X) - 10], X[len(X) - 10 :]
Y_train, Y_test = Y[: len(Y) - 10], Y[len(Y) - 10 :]

X_train_tensors, X_test_tensors = Variable(torch.Tensor(X_train.values)), Variable(torch.Tensor(X_test.values))
Y_train_tensors, Y_test_tensors = Variable(torch.Tensor(Y_train.values)), Variable(torch.Tensor(Y_test.values))

X_train_tensors_f = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
X_test_tensors_f = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))

num_epochs = 1000
lr = 0.0001

input_size = 7
hidden_size = 2
num_layers = 1

num_classes = 1

model = nn.LSTM(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in tqdm(range(num_epochs)):
    outputs = model.forward(X_train_tensors_f)
    optimizer.zero_grad()
    loss = criterion(outputs, Y_train_tensors)
    loss.backward()

    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, loss: {loss.item()}")