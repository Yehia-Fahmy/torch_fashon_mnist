import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

batch_size = 128
num_workers = 8

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size, True, num_workers=num_workers)
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size, True, num_workers=num_workers)
classes = trainset.classes

print("Dataset Information")
print("trainset dimensions:")
print(f"number of train samples: {len(trainset)}")
print(f"number of test samples: {len(testset)}")
print(f"image size: {trainset[0][0].shape}")
print(f"number of classes: {len(classes)}, and they are: {classes}")

class FasionClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 3)
        # [16, 26, 26]
        # with mp ->  [16, 13, 13]
        self.conv2 = nn.Conv2d(16, 16, 3)
        # [16, 24, 24]
        # with mp -> # [16, 5, 5]
        self.conv3 = nn.Conv2d(16, 16, 3)
        # [16, 22, 22]

        self.fc1 = nn.Linear(16 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)

        self.mp = nn.MaxPool2d(2, 2)
        # half the size

        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.mp(x)
        x = self.relu(self.conv2(x))
        x = self.mp(x)
        x = self.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        # print(x.shape)
        # exit()
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = FasionClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
number_of_epochs = 10

for epoch in range(number_of_epochs):
    print(f"Epoch {epoch+1}:")
    # Train
    running_loss, total_samples, correct_samples = 0.0, 0, 0
    model.train()
    for x, y in trainloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        # Loss
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # Accuracy
        _, predicted = torch.max(y_hat, 1)
        total_samples += y.size(0)
        correct_samples += (predicted == y).sum().item()

    
    print(f"Training Loss: {(running_loss / len(trainloader)):.4f} Accuracy: {(100 * correct_samples / total_samples):.2f}%")

    # Test
    running_loss, total_samples, correct_samples = 0.0, 0, 0
    model.eval()
    for (x, y) in testloader:
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        # Loss
        loss = criterion(y_hat, y)
        running_loss += loss.item()
        # Accuracy
        _, predicted = torch.max(y_hat, 1)
        total_samples += y.size(0)
        correct_samples += (predicted == y).sum().item()
    print(f"Testing Loss: {(running_loss / len(trainloader)):.4f} Accuracy: {(100 * correct_samples / total_samples):.2f}%")
