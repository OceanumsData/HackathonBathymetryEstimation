import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
from functools import partial
from collections import OrderedDict
from dataclasses import dataclass
import time
import numpy as np

from config import USE_RAW


class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size

conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels =  in_channels, out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(OrderedDict(
        {
            'conv' : nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            'bn' : nn.BatchNorm2d(self.expanded_channels)
        })) if self.should_apply_shortcut else None
    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels
def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs),
                          'bn': nn.BatchNorm2d(out_channels) }))


class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            nn.Dropout2d(0.25),
            activation(),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
            nn.Dropout2d(0.25),
            activation(),
        )


class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion,
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )
    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by increasing different layers with increasing features.
    """
    def __init__(self, in_channels=4, blocks_sizes=[64, 128, 256, 512], deepths=[2,2,2,2],
                 activation=nn.ReLU, block=ResNetBasicBlock, *args,**kwargs):
        super().__init__()
        self.blocks_sizes = blocks_sizes
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation,
                        block=block,  *args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion,
                          out_channels, n=n, activation=activation,
                          block=block, *args, **kwargs)
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]
        ])
    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x


class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features, n_classes)
    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x



class ResNet(nn.Module):
    
    def __init__(self, device, in_channels=(8 if USE_RAW else 4), n_classes=1, deepths=[2,2,2,2]):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.deepths = deepths
        self.device = device
        self.reset()
        
    def reset(self):
        self.encoder = ResNetEncoder(self.in_channels, self.deepths)
        self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, self.n_classes)
        self.to(self.device)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x.flatten()
    
    def fit(self, train_dataloader, test_dataloader, n_epochs, print_frequency, optimizer=None, criterion=nn.MSELoss()):
        
        while True:
        
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
            if np.mean(train_losses) < 15:
                break
            
            self.reset()
            
    def predict(self, dataloader):
        predictions = []
        for it, batch in enumerate(dataloader):
            out = list(self(batch["image"].to(self.device)).cpu().detach().numpy())
            predictions += out
        return predictions
    