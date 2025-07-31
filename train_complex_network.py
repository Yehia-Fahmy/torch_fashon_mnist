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

# Improved data augmentation
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

batch_size = 64  # Smaller batch size for better generalization
num_workers = 4

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size, True, num_workers=num_workers)
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size, True, num_workers=num_workers)
classes = trainset.classes

print("Dataset Information")
print("trainset dimensions:")
print(f"number of train samples: {len(trainset)}")
print(f"number of test samples: {len(testset)}")
print(f"image size: {trainset[0][0].shape}")
print(f"number of classes: {len(classes)}, and they are: {classes}")

class ImprovedFashionClassifier(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        
        # Convolutional layers with increasing channels
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  # 32 channels, padding to maintain size
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # MaxPool layers
        self.mp = nn.MaxPool2d(2, 2)
        
        # Global Average Pooling instead of flattening
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers with dropout
        self.fc1 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # First conv block: 28x28 -> 14x14
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.mp(x)
        
        # Second conv block: 14x14 -> 7x7
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.mp(x)
        
        # Third conv block: 7x7 -> 3x3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.mp(x)
        
        # Global average pooling: 3x3 -> 1x1
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

model = ImprovedFashionClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Added weight decay
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
number_of_epochs = 20  # More epochs for better training

# Training history tracking
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
best_test_accuracy = 0.0

for epoch in range(number_of_epochs):
    print(f"Epoch {epoch+1}/{number_of_epochs}")
    
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

    train_loss = running_loss / len(trainloader)
    train_accuracy = 100 * correct_samples / total_samples
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    
    print(f"Training Loss: {train_loss:.4f} Accuracy: {train_accuracy:.2f}%")

    # Test
    running_loss, total_samples, correct_samples = 0.0, 0, 0
    model.eval()
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            # Loss
            loss = criterion(y_hat, y)
            running_loss += loss.item()
            # Accuracy
            _, predicted = torch.max(y_hat, 1)
            total_samples += y.size(0)
            correct_samples += (predicted == y).sum().item()
    
    test_loss = running_loss / len(testloader)
    test_accuracy = 100 * correct_samples / total_samples
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    
    print(f"Testing Loss: {test_loss:.4f} Accuracy: {test_accuracy:.2f}%")
    
    # Learning rate scheduling
    scheduler.step(test_loss)
    
    # Save best model
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        torch.save(model.state_dict(), 'best_fashion_model.pth')
        print(f"New best model saved! Accuracy: {best_test_accuracy:.2f}%")
    
    print("-" * 50)

print(f"Training completed! Best test accuracy: {best_test_accuracy:.2f}%")
