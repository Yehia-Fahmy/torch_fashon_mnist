import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BATCH_SIZE=64
NUM_WORKERS=0
EPOCHS=20

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
        self.conv1 = nn.Conv2d(1, 4, 3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(4, 8, 3)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(8, 8, 3)
        self.relu3 = nn.ReLU()
        self.linear = nn.Linear(8 * 22 * 22, 10)
    
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x

# initialize network, loss function and optimizer
network = MyNetwork().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.parameters())

# manually specify training loop
for epoch in range(EPOCHS):
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
print('Finished Training')

