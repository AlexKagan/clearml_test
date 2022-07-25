import os
import psutil
import clearml
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

class CifarTrain:
    def __init__(self, config) -> None:
        self.config = config
        self.config["device"] = torch.device('cuda' if (torch.cuda.is_available() and self.config["cuda"]) else 'cpu')
        self.main()

    
    def load_data(self):
        transform = transforms.Compose([transforms.ToTensor()])

        train_set = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.config["batch_size"],
                                          shuffle=True, num_workers=2)
        val_set = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)
        self.val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.config["batch_size"],
                                          shuffle=False, num_workers=2)
        self.classes = [str(i) for i in range(10)]
        
    def main(self):
        self.load_data()
        dataiter = iter(self.train_loader)
        images, labels = dataiter.next()
        torch.max(images)