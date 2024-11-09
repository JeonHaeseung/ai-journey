import os
from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

from cnn_pytorch import *


# GPU
GPU_ID = "0"

# Log
CURRENT_DIR = os.path.dirname(__file__)
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_DIR = os.path.join(CURRENT_DIR, "log")
TB_LOG_PATH = os.path.join(LOG_DIR, f"alexnet_{TIMESTAMP}")

# Dataset is (channel, width, height) == (3, 227, 227)
DATA_DIR = os.path.join(CURRENT_DIR, "data")
IMAGE_SIZE = 32        # Original AlexNet's image size is 227
NUM_CLASSES = 10

# Hyperparameter
BATCH_SIZE = 128
EPOCHS = 10
LR = 0.0001

def set_dirs():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def set_device():
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def set_dataset():
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])

    train_dataset = CIFAR10(root=DATA_DIR,
                        train=True,
                        download=True,
                        transform=transform)

    test_dataset = CIFAR10(root=DATA_DIR,
                        train=False,
                        download=True,
                        transform=transform)

    train_loader = DataLoader(dataset=train_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=4)

    test_loader = DataLoader(dataset=test_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        num_workers=4)
    
    return train_loader, test_loader


def train_model(model, train_loader, criterion, optimizer, epochs, writer, device):
    model.train()
    
    for epoch in range(epochs):
        running_train_loss = 0.0
        correct = 0
        total = 0

        # Progress bar
        with tqdm(train_loader, unit='batch') as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{epochs}")

            for batch, (inputs, labels) in enumerate(tepoch):
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item()

                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Update with 100 batches
                if batch % 100 == 99:
                    accuracy = 100 * correct / total
                    tepoch.set_postfix(loss=(running_train_loss / 100), accuracy=accuracy)

                    # TensorBoard
                    global_step = epoch * len(train_loader) + batch
                    writer.add_scalar('Loss/train', running_train_loss / 100, global_step)
                    writer.add_scalar('Accuracy/train', accuracy, global_step)

                    running_train_loss = 0.0
                    correct = 0
                    total = 0


def test_model(model, test_loader, device, writer):
    model.eval()
    correct = 0
    total = 0

    # No gradient update while testing
    with torch.no_grad():
        for (inputs, labels) in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    writer.add_scalar('Accuracy/test', accuracy)


def set_model(device):
    model = AlexNet(num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    model = model.to(device)
    return model, criterion, optimizer


def run_experiment():
    set_dirs()
    device = set_device()
    train_loader, test_loader = set_dataset()
    model, criterion, optimizer = set_model(device)
    
    # check with terminal command: tensorboard --logdir TB_LOG_PATH
    # You can see the logs in localhost:6006
    writer = SummaryWriter(log_dir=TB_LOG_PATH)

    train_model(model, train_loader, criterion, optimizer, EPOCHS, writer, device)
    test_model(model, test_loader, device, writer)


if __name__ == "__main__":
    run_experiment()