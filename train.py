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

# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Argument parser
def get_args():
    parser = argparse.ArgumentParser(description='Multi-classification PyTorch Trainer')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'fashionmnist'], help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save-plots', action='store_true', help='Save plots as images')
    return parser.parse_args()

# Dataset loader
def get_dataloaders(dataset, batch_size):
    if dataset == 'cifar10':
        num_classes = 10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        classes = trainset.classes
    elif dataset == 'fashionmnist':
        num_classes = 10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        classes = trainset.classes
    else:
        raise ValueError('Unsupported dataset')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader, num_classes, classes

# Simple CNN model
def get_model(dataset, num_classes):
    if dataset == 'cifar10':
        in_channels = 3
    else:
        in_channels = 1
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 8 * 8 if in_channels == 3 else 64 * 7 * 7, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x
    return SimpleCNN().to(device)

# Training and validation
def train_and_validate(model, trainloader, testloader, epochs, lr, classes, save_plots):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs} [Train]'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in tqdm(testloader, desc=f'Epoch {epoch+1}/{epochs} [Val]'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}')
    # Visualization
    plot_curves(train_losses, val_losses, train_accs, val_accs, save_plots)
    plot_confusion_matrix(all_labels, all_preds, classes, save_plots)
    print(classification_report(all_labels, all_preds, target_names=classes))
    plot_sample_predictions(model, testloader, classes, save_plots)

def plot_curves(train_losses, val_losses, train_accs, val_accs, save):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.subplot(1,2,2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.legend()
    plt.title('Accuracy Curves')
    plt.tight_layout()
    if save:
        plt.savefig('training_curves.png')
    plt.show()

def plot_confusion_matrix(labels, preds, classes, save):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    if save:
        plt.savefig('confusion_matrix.png')
    plt.show()

def plot_sample_predictions(model, testloader, classes, save):
    model.eval()
    images, labels = next(iter(testloader))
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, preds = outputs.max(1)
    images = images.cpu().numpy()
    plt.figure(figsize=(12,6))
    for i in range(8):
        plt.subplot(2,4,i+1)
        img = images[i].transpose(1,2,0) if images.shape[1] == 3 else images[i][0]
        if images.shape[1] == 1:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow((img * 0.5) + 0.5)
        plt.title(f'True: {classes[labels[i]]}\nPred: {classes[preds[i]]}')
        plt.axis('off')
    plt.tight_layout()
    if save:
        plt.savefig('sample_predictions.png')
    plt.show()

def main():
    args = get_args()
    trainloader, testloader, num_classes, classes = get_dataloaders(args.dataset, args.batch_size)
    model = get_model(args.dataset, num_classes)
    train_and_validate(model, trainloader, testloader, args.epochs, args.lr, classes, args.save_plots)

if __name__ == '__main__':
    main()

