---
title: "Facial Recognition Using One-shot Learning"
date: 2022-12-18T21:36:21+02:00
#draft: true
author: "Yuval"

resources:
  -name: "featured-image"
  src: "featured-image.jpg"

tags: ["Python", "Siamese Neural Networks", "Deep Learning"]
categories: ["Neural Networks"]
---


### Implementation

#### Imports
First, there's a need to import all libraries that we will use. NumPy (np) and Pandas (pd) are fundamental for numerical computations and data manipulation, respectively. The tqdm library provides progress bars for better visualization during loops or computations, while the time module allows us to work with time-related functions.

We also import PyTorch (torch) for building and training deep learning models. OpenCV (cv2) is essential for image processing tasks.
```Python
import numpy as np
import pandas as pd

from tqdm import tqdm
import time

import torch
import cv2 as cv
from zipfile import ZipFile
import os

from torch.utils.tensorboard import SummaryWriter

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import matplotlib.pyplot as plt
```

#### Class for image pair processing

This code defines a custom dataset class named TwoImageDataset for working with a dataset containing pairs of images and corresponding labels.

```Python
# Images does not stored in memory, we read the image only when we use __getitem__

class TwoImageDataset(Dataset):
    def __init__(self, image_dir):
        """
        Initializes the dataset object with the directory containing image pairs.

        Args:
            image_dir (str): Directory containing image pairs.
        """
        self.image_dir = image_dir
        # Read the contents of the file and store them as samples
        with open(self.image_dir) as file:
            # Skip the header line (assuming it contains column names)
            self.samples = [line.rstrip('\n').split('\t') for line in file][1:]

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Fetches a sample from the dataset at a given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the preprocessed images and their corresponding label.
        """
        return self.read_images(self.samples[idx])

    def path(self, name, pic_num):
        """
        Constructs the file path for an image given its name and picture number.

        Args:
            name (str): Name of the image.
            pic_num (str): Picture number of the image.

        Returns:
            str: File path of the image.
        """
        return f'/content/lfw2/lfw2/{name}/{name}_{pic_num.zfill(4)}.jpg'

    def read_image(self, name, pic_num):
        """
        Reads and preprocesses a single image specified by its name and picture number.

        Args:
            name (str): Name of the image.
            pic_num (str): Picture number of the image.

        Returns:
            tuple: A tuple containing the image name and a PyTorch tensor representing the preprocessed image.
        """
        img = cv.imread(self.path(name, pic_num), 0)
        img = cv.resize(img, (105, 105)) / 255
        img = np.expand_dims(img, axis=0)
        return name, torch.tensor(img).type(torch.float)

    def read_images(self, sample):
        """
        Reads and preprocesses a pair of images specified by the sample.

        Args:
            sample (list): A list containing image names and picture numbers.

        Returns:
            tuple: A tuple containing the preprocessed images and their corresponding label.
        """
        if len(sample) == 3:
            img1 = self.read_image(sample[0], sample[1])
            img2 = self.read_image(sample[0], sample[2])
            label = torch.tensor([1.])
            return (img1, img2, label)
        else:
            img1 = self.read_image(sample[0], sample[1])
            img2 = self.read_image(sample[2], sample[3])
            label = torch.tensor([0.])
            return (img1, img2, label)
```

#### Building the Siamese Neural Network

This code defines a Siamese Neural Network (SNN) architecture using PyTorch for image comparison tasks
This Siamese NN consists of convolutional layers followed by fully connected layers. The convolutional layers extract features from input images, which are then passed through fully connected layers to produce a vector representation. Finally, the absolute difference between the vector representations of two input images is computed, and the result is passed through a fully connected layer to obtain the final output. The init_weights method initializes the weights of convolutional and linear layers using a normal distribution.


```Python
class SiameseNN(nn.Module):
    def __init__(self):
        """
        Initializes the Siamese Neural Network architecture.
        """
        super(SiameseNN, self).__init__()

        # Define the convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 10), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 7), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 4), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 4), nn.ReLU()
        )

        # Define the fully connected layers for vector representation
        self.to_vector = nn.Sequential(
            nn.Linear(9216, 4096), nn.Sigmoid()
        )

        # Define the fully connected layer for output
        self.to_output = nn.Sequential(
            nn.Linear(4096, 1), nn.Sigmoid()
        )

        # Initialize weights of convolutional and linear layers
        self.conv.apply(self.init_weights)
        self.to_vector.apply(self.init_weights)
        self.to_output.apply(self.init_weights)

    def forward(self, img1, img2):
        """
        Forward pass of the Siamese Neural Network.

        Args:
            img1 (torch.Tensor): Input image 1.
            img2 (torch.Tensor): Input image 2.

        Returns:
            torch.Tensor: Result of the forward pass.
        """
        # Forward pass through convolutional layers
        input1 = self.conv(img1)
        input1 = input1.view(input1.shape[0], -1)
        input1 = self.to_vector(input1)

        input2 = self.conv(img2)
        input2 = input2.view(input2.shape[0], -1)
        input2 = self.to_vector(input2)

        # Compute absolute difference and pass through output layer
        result = self.to_output(torch.abs(input1 - input2))

        return result

    def init_weights(self, m):
        """
        Initializes weights of convolutional and linear layers with a normal distribution.

        Args:
            m (nn.Module): Module to initialize weights.
        """
        # Initialize weights of convolutional layers with normal distribution
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight, 0, 0.01)
            m.bias.data.normal_(0, 0.01)
        # Initialize weights of linear layers with normal distribution
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, 0, 0.5)
```


#### Train

This code defines a train function used for training a neural network model. It iterates over batches of data, computes the model's predictions, calculates the loss using the specified criterion, performs L2 regularization, backpropagates the loss, and updates the model parameters using the optimizer.

```Python
def train(dataloader, model, criterion, optimizer, epoch, verbose=True):
    """
    Trains the model.

    Args:
        dataloader (DataLoader): DataLoader for training data.
        model (nn.Module): Model to be trained.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        epoch (int): Current epoch.
        verbose (bool, optional): Whether to print training progress. Default is True.

    Returns:
        float: Accuracy of the model.
    """
    model.train()
    for step, (x1, x2, y) in enumerate(dataloader):
        running_loss = 0.0
        x1[1], x2[1], y = x1[1].to(device), x2[1].to(device), y.to(device)

        # Compute prediction error
        pred = model(x1[1], x2[1])
        loss = criterion(pred, y)

        # L2 regularization
        l2_lambda = 0.0000001
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        loss = loss + l2_lambda * l2_norm

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (step + 1) % 5 == 0:
            running_loss /= 5

            preds = (pred > 0.5).type(torch.float)
            accuracy = (preds == y).type(torch.float).sum().item() / len(y)
            if verbose:
                print(f'Step: {step + 1} Loss: {running_loss} Accuracy: {accuracy}')

            # Log the running loss
            writer.add_scalar('train loss',
                              running_loss / 1000,
                              epoch * len(dataloader) + step)

            writer.add_scalar('train accuracy',
                              accuracy / 1000,
                              epoch * len(dataloader) + step)
    return accuracy

```
