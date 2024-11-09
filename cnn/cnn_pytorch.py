import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        # ====== 1st Conv ======
        # (width + 2*padding - kernel)/stride + 1
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=1, padding=5)      # original stride==4, padding==0
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # ====== 2nd Conv ======
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # ====== 3rd Conv ======
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)

        # ====== 4th Conv ======
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)

        # ====== 5th Conv ======
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=2)    # original padding==1
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        # ====== 1st Dense ======
        self.fc1 = nn.Linear(256 * 4 * 4, 4096) # original is 256 * 6 * 6
        self.dropout1 = nn.Dropout(0.5)

        # ====== 2nd Dense ======
        self.fc2 = nn.Linear(4096, 4096)
        self.dropout2 = nn.Dropout(0.5)

        # ====== 3rd Dense ======
        self.fc3 = nn.Linear(4096, num_classes)


    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool3(torch.relu(self.conv5(x)))
        x = x.view(-1, 256 * 4 * 4)             # original is 256 * 6 * 6
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x