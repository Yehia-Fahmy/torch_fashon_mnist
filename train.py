import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BATCH_SIZE=128
NUM_WORKERS=0
EPOCHS=10

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def get_fashion_mnist_label(label_id):
    label_map = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot"
    }
    return label_map.get(label_id, "Unknown")


# Define transforms for normalization and tensor conversion
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

testset = torchvision.datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)
        self.finalLinear = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = self.finalLinear(x)
        return x

# initialize network, loss function and optimizer
network = MyNetwork().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.parameters())

# manually specify training loop
for epoch in range(EPOCHS):
    # sets network in training mode
    network.train()
    print(f"training epoch: {epoch}")
    # track loss for this epoch just for tracking purposes
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # print(f"training batch: {i}")
        # for each batch split inputs and lables
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # need to reset the gradient of the optimizer every batch
        optimizer.zero_grad()
        # forward pass
        output = network(inputs)
        # find loss and the gradient
        loss = criterion(output, labels)
        loss.backward()
        # update weights
        optimizer.step()
        # update running loss
        running_loss += loss.item()
    print(f"Epoch {epoch} Loss: {running_loss}")

    # evaluate on training set
    network.eval()
    correct_train = 0
    total_train = 0
    with torch.no_grad():
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = network(inputs)
            _, preds = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (preds == labels).sum().item()
    train_acc = 100 * correct_train / total_train

    # evaluate on test set
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = network(inputs)
            _, preds = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (preds == labels).sum().item()
    test_acc = 100 * correct_test / total_test

    print(f"Epoch {epoch} Train Accuracy: {train_acc:.2f}% Test Accuracy: {test_acc:.2f}%")

print('Finished Training')

