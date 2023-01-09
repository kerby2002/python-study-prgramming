import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score


class DigitDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.targets[item]

class LinearNetwork(nn.Module):
    
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_size)
        self.loss = F.mse_loss
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x 

data = load_digits()
x = data["data"]
y = data["target"]
Y = np.zeros((len(y), 10))
indices = np.arange(len(y))

Y[indices, y] = 1
batch_size = 64
X_train, X_test, Y_train, Y_test = train_test_split(x,Y,train_size=0.7)
train_dataset = DigitDataset(data=X_train, targets=Y_train)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = DigitDataset(data=X_test, targets=Y_test)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

lin_net = LinearNetwork(input_size = 64, output_size = 10)
optimizer = optim.Adam(params=lin_net.parameters(),lr=0.001)
epochs = 10

for epoch in range(epochs):
    if epoch % 1 == 0:
        lin_net.eval()
        correct = 0
        
        for x,y in test_dataloader:
            x = x.to(torch.float)
            y = y.to(torch.float)
            y_hat = lin_net(x)
            y_hat = torch.squeeze(y_hat,dim=1)
            choices = torch.argmax(y_hat,dim=1)
            indices = np.arange(len(x))
            y_hat = torch.zeros_like(y_hat)
            y_hat[indices,choices] = 1
            score = accuracy_score(y_hat,y)
            correct += score*len(x)
        
        print(correct/len(test_dataset))
        lin_net.train()

    for x, y in train_dataloader:
        x = x.to(torch.float)
        y = y.to(torch.float)
        #zero out the gradients
        
        optimizer.zero_grad()
        y_hat = lin_net(x)
        y_hat = torch.squeeze(y_hat,dim=1)
        loss = lin_net.loss(y_hat,y)
        

        loss.backward()
        optimizer.step()
    