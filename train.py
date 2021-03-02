import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
from functools import partial
from dataclasses import dataclass

from model import ResNet
from dataset import HackathonDataset

batchSize=256
model = ResNet(4,1, deepths=[2,2,2,2]) #ResNet 18

#Normalizing inputs to [0,1]

def trans(img, m= -1, M =2):
    img = (img - m)/M
    return img

dataset = HackathonDataset("../input/bathymetry-estimation/mixed_train.csv", "../input/bathymetry-estimation/", transform = trans)
dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(DEVICE)

batch_losses = []
epochs = 25

for i in range(epochs):
    print("epochs : " + str(i))
    for it, batch in enumerate(dataloader):

            # Reset gradients
            optimizer.zero_grad()

            # Forward propagation through the network
            out = model(batch["image"].to(DEVICE))

            # Calculate the loss
            loss = torch.sqrt(criterion(out.flatten(), batch["z"].to(DEVICE)))  # We take square root because RMSE is the competition's metric

            # Track batch loss
            batch_losses.append(loss.item())

            # Backpropagation
            loss.backward()

            # Update the parameters
            optimizer.step()

            #=====Printing part======
            if it%100 == 0:
                print(f"Number of sample viewed : {it*batchSize}")
    valset = HackathonDataset("../input/bathymetry-estimation/mixed_validation.csv", "../input/bathymetry-estimation/")
    val_loader = DataLoader(valset, batch_size=batchSize, shuffle=True, num_workers=2)
    val_losses=[]
    for it, batch in enumerate(val_loader):

        # Forward propagation through the network
        out = model(batch["image"].to(DEVICE))

        # Calculate the loss
        loss = torch.sqrt(criterion(out.flatten(), batch["z"].to(DEVICE)))  # We take square root because RMSE is the competition's metric

        # Track batch loss
        val_losses.append(loss.item())
        if it%100==0:
            print("*")
    print("Val loss : " + str(np.mean(val_losses)))
    print("Train loss : " + str(np.mean(batch_losses)))
    batch_losses=[]
