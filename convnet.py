import torch
import torch.nn as nn
import time
import numpy as np

from config import USE_RAW


class ConvNet(nn.Module):
    """Used Network"""

    def __init__(self, device, use_raw = USE_RAW):
        self.use_raw = use_raw
        super(ConvNet, self).__init__()
        self.device = device
        self.reset()
    
    def reset(self):
        self.conv1 = nn.Conv2d(8 if self.use_raw else 4, 32, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
#         self.conv3 = nn.Conv2d(32, 256, 3, padding=1)
#         self.relu3 = nn.ReLU()
#         self.pool3 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(6400, 1024)
#        self.dropout = nn.Dropout(0.1)
        self.relu4 = nn.ReLU()
#         self.fc2 = nn.Linear(1024, 1024)
#         self.relu5 = nn.ReLU()
        self.fc3 = nn.Linear(1024, 1)
        self.relu6 = nn.ReLU()
        self.to(self.device)

    def forward(self, x):
        """Given a tensor X of shape (Batch_size, C_in, H, W), compute the output tensor, of shape (Batch_size, )"""
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
#         x = self.conv3(x)
#         x = self.relu3(x)
#         x = self.pool3(x)
        x = x.flatten(start_dim=1)  # Flatten the 3 last dimensions, keep the 1 dimension (batch_size)
        x = self.fc1(x)
        x = self.relu4(x)
#         x = self.fc2(x)
#         x = self.relu5(x)
        x = self.fc3(x)
        x = self.relu6(x)
        x = x.flatten()  # The output dimension should be (Batch_size, ) and not (Batch_size, 1) 
        return x
    
    def fit(self, train_dataloader, test_dataloader, n_epochs, print_frequency, optimizer=None, criterion=nn.MSELoss()):
        
#        self.train()
                
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters())
        
        train_losses = []
        test_losses = []
        
        for epoch in range(n_epochs):
            
            timer = time.time()
            
            # Print Epoch
            print(f"Epoch {epoch + 1}/{n_epochs}")
            
            # Training loop
            for it, batch in enumerate(train_dataloader):
                        
                # Reset gradients
                optimizer.zero_grad()
                
                # Forward propagation through the network
                out = self(batch["image"].to(self.device))

                # Calculate the loss
                loss = torch.sqrt(criterion(out, batch["z"].to(self.device)))  # We take square root because RMSE is the competition's metric

                # Track batch loss
                train_losses.append(loss.item())

                # Backpropagation
                loss.backward()

                # Update the parameters
                optimizer.step()

                #=====Printing part======
                if (it+1)%(len(train_dataloader) // print_frequency) == 0:
                    print(f"Number of batches viewed : {it}")
                    print(f"Current training loss : {np.mean(train_losses[-len(train_dataloader)//print_frequency:-1])}")

                    # Validation loop
                    for it, batch in enumerate(test_dataloader):

                        # Forward propagation through the network
                        out = self(batch["image"].to(self.device))

                        # Calculate the loss
                        loss = torch.sqrt(criterion(out, batch["z"].to(self.device)))  # We take square root because RMSE is the competition's metric

                        # Track batch loss
                        test_losses.append(loss.item())
            
                    print(f"Current validation loss : {np.mean(test_losses[-int(len(test_dataloader)*0.8):-1])}")
            
            print(f"The epoch took {time.time() - timer: .2f} seconds")
            
        return np.mean(test_losses[-(len(test_losses)//10):-1])
                        
    def predict(self, dataloader):
        
#        self.eval()
        
        predictions = []
        for it, batch in enumerate(dataloader):
            out = list(self(batch["image"].to(self.device)).cpu().detach().numpy())
            predictions += out
        return predictions