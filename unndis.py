# Nathan HK
# 2024-09-05

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import jarnbra
import landbjartur

"""
Command-line arguments:
1. Directory for images
2. Number of epochs

PEP-8 compliant.
"""

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


class VGLikan(nn.Module):
    def __init__(self):
        super(VGLikan, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1,
                               dtype=torch.float16)    # Output: 512x512x16
        self.pool = nn.MaxPool2d(2, 2)  # Output size halved
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1,
                               dtype=torch.float16)   # Output: 256x256x32
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,
                               dtype=torch.float16)   # Output: 128x128x64
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1,
                               dtype=torch.float16)  # Output: 64x64x128
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1,
                               dtype=torch.float16)  # Output: 32x32x256
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1,
                               dtype=torch.float16)  # Output: 16x16x512
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1,
                               dtype=torch.float16)  # Output: 8x8x512
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1,
                               dtype=torch.float16)  # Output: 4x4x512

        # Fully connected layers
        self.fc1 = nn.Linear(2 * 2 * 512, 512, dtype=torch.float16)
        self.fc2 = nn.Linear(512, 128, dtype=torch.float16)
        self.fc3 = nn.Linear(128, 1, dtype=torch.float16)

    def forward(self, x):
        # Input x: (batch_size, 1, 1024, 1024)
        x = F.relu(self.conv1(x))  # Output: (batch_size, 16, 512, 512)
        x = self.pool(x)           # Output: (batch_size, 16, 256, 256)

        x = F.relu(self.conv2(x))  # Output: (batch_size, 32, 256, 256)
        x = self.pool(x)           # Output: (batch_size, 32, 128, 128)

        x = F.relu(self.conv3(x))  # Output: (batch_size, 64, 128, 128)
        x = self.pool(x)           # Output: (batch_size, 64, 64, 64)

        x = F.relu(self.conv4(x))  # Output: (batch_size, 128, 64, 64)
        x = self.pool(x)           # Output: (batch_size, 128, 32, 32)

        x = F.relu(self.conv5(x))  # Output: (batch_size, 256, 32, 32)
        x = self.pool(x)           # Output: (batch_size, 256, 16, 16)

        x = F.relu(self.conv6(x))  # Output: (batch_size, 512, 16, 16)
        x = self.pool(x)           # Output: (batch_size, 512, 8, 8)

        x = F.relu(self.conv7(x))  # Output: (batch_size, 512, 8, 8)
        x = self.pool(x)           # Output: (batch_size, 512, 4, 4)

        x = F.relu(self.conv8(x))  # Output: (batch_size, 512, 4, 4)
        x = self.pool(x)           # Output: (batch_size, 512, 2, 2)

        x = x.view(-1, 2 * 2 * 512)  # Flatten the tensor
        x = F.relu(self.fc1(x))      # Fully connected layer
        x = F.relu(self.fc2(x))      # Fully connected layer
        x = self.fc3(x)              # Output layer
        x = torch.sigmoid(x)         # Sigmoid for binary classification

        return x


def train_VGLikan(X_train, X_test, y_train, y_test, epochs):
    byrjun = time.time()
    # Initialize model
    likan = VGLikan().to(device)

    # Define a loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.SGD(likan.parameters(), lr=5e-4, momentum=0.25)

    train_loss = []
    test_loss = []

    for e in range(epochs):
        likan.eval()
        with torch.no_grad():
            rl = 0.0
            for i in range(len(X_train)):
                y_pred = likan(X_train[i])
                loss = criterion(y_pred, y_train[i])
                rl += loss.item()
            train_loss.append(rl / len(X_train))
            rl = 0.0
            for i in range(len(X_test)):
                y_pred = likan(X_test[i])
                loss = criterion(y_pred, y_test[i])
                rl += loss.item()
            test_loss.append(rl / len(X_test))
        torch.mps.empty_cache()

        likan.train()
        for i in range(len(X_train)):
            y_pred = likan(X_train[i])
            loss = criterion(y_pred, y_train[i])
            if np.isnan(loss.item()):
                print('ERROR: NaN', i)
                print(train_loss)
                print(test_loss)
                break

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            total_norm = 0
            for p in likan.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            if np.isnan(total_norm):
                nn.utils.clip_grad_norm_(likan.parameters(), 100.0)
                continue
            nn.utils.clip_grad_norm_(likan.parameters(), 100.0)
            optimizer.step()
            torch.mps.synchronize()
            if i % 10 == 0:
                torch.mps.empty_cache()

        torch.mps.empty_cache()

        # Print loss for the current epoch
        print(f'Epoch [{e+1}/{epochs}], Time: {time.time() - byrjun}')

    likan.eval()
    with torch.no_grad():
        rl = 0.0
        for i in range(len(X_train)):
            y_pred = likan(X_train[i])
            loss = criterion(y_pred, y_train[i])
            rl += loss.item()
        train_loss.append(rl / len(X_train))
        rl = 0.0
        for i in range(len(X_test)):
            y_pred = likan(X_test[i])
            loss = criterion(y_pred, y_test[i])
            rl += loss.item()
        test_loss.append(rl / len(X_test))
    torch.mps.empty_cache()

    print('Train loss:', train_loss[-1])
    print('Test loss:', test_loss[-1])

    plt.plot(train_loss, label='Train')
    plt.plot(test_loss, label='Test')
    plt.ylim(bottom=0)
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    thresh = 0.5
    likan.eval()
    with torch.no_grad():
        train = [0, 0, 0]  # TP, FP, FN
        test = [0, 0, 0]  # TP, FP, FN
        for i in range(len(X_train)):
            y_pred = likan(X_train[i])
            if y_pred.item() >= thresh and y_train[i].item() == 1:
                train[0] += 1
            elif y_pred.item() >= thresh and y_train[i].item() == 0:
                train[1] += 1
            elif y_pred.item() < thresh and y_train[i].item() == 1:
                train[2] += 1
        for i in range(len(X_test)):
            y_pred = likan(X_test[i])
            if y_pred.item() >= thresh and y_test[i].item() == 1:
                test[0] += 1
            elif y_pred.item() >= thresh and y_test[i].item() == 0:
                test[1] += 1
            elif y_pred.item() < thresh and y_test[i].item() == 1:
                test[2] += 1
    print('Train precision:', train[0] / (train[0] + train[1]))
    print('Train recall:', train[0] / (train[0] + train[2]))
    print('Test precision:', test[0] / (test[0] + test[1]))
    print('Test recall:', test[0] / (test[0] + test[2]))


if __name__ == '__main__':
    hnitlisti = jarnbra.getCoordList(sys.argv[1])
    X_gogn, y_gogn = landbjartur.getRoadData('U', sys.argv[1], hnitlisti)
    X_train, X_test, y_train, y_test = train_test_split(X_gogn, y_gogn)
    train_VGLikan(X_train, X_test, y_train, y_test, int(sys.argv[2]))
