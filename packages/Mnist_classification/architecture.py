from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleClassifier(nn.Module):
    """
    Classifier for Mnist dataset input image size (1, 28, 28)
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.fc1 = nn.Linear(7 * 7 * 32, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

x = torch.rand((1,1,28,28))
net = SimpleClassifier()
x1 = net(x)
print(x1.shape)