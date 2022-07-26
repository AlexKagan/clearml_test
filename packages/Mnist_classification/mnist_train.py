import os
import psutil
import clearml
from clearml import Task
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import datetime
from Mnist_classification import architecture
from utils import IO


torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

class MnistTrain:
    def __init__(self, config) -> None:
        self.task = Task.init(project_name="Clearml_test", task_name=f"Mnist_exp", reuse_last_task_id=True,
                         auto_resource_monitoring=True, auto_connect_streams=True)
        self.logger = self.task.get_logger()
        self.config = config
        self.best_metric = 0
        self.config["device"] = torch.device('cuda' if (torch.cuda.is_available() and self.config["cuda"]) else 'cpu')
        self.device = self.config["device"]
        print(f"Device: {self.device}")
        self.run()

    
    def load_data(self):
        transform = transforms.Compose([transforms.ToTensor()])

        train_set = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.config["batch_size"],
                                          shuffle=True, num_workers=self.config["num_worker"])
        val_set = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)
        self.val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.config["batch_size"],
                                          shuffle=False, num_workers=self.config["num_worker"])
        self.classes = [str(i) for i in range(10)]

    def upload_net(self):
        self.net = architecture.SimpleClassifier()
        self.net.to(self.device)
    
    def train_net(self):
        print("starting train, nr of epochs: ", self.config["num_epochs"])
        self.criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.net.parameters(), lr=self.config["lr"], weight_decay=self.config['weight_dec'])
        for self.epoch in range(self.config["num_epochs"]):
            t0 = time.time()
            running_loss = 0
            correct, total = 0, 0
            self.net.train()
            for i, data in enumerate(self.train_loader):
                inputs, labels = [i.to(self.device) for i in data]
                optimizer.zero_grad()
                outputs = self.net(inputs)
                pred = torch.argmax(outputs, 1)
                loss = self.criterion(outputs,labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                total += labels.size(0)
                correct += (pred == labels).sum().item()

            train_loss = running_loss/(i+1)
            train_accuracy = correct/total
            print(f"Epoch: {self.epoch}, time: {time.time() - t0}, train_loss: {round(train_loss, 4)}, Train_accuracy: {round(train_accuracy, 4)}")
            self.logger.report_scalar(title='Train vs Val loss',
                                        series='Train_loss', value=train_loss, iteration=self.epoch)
            self.logger.report_scalar(title='Train vs Val accuracy',
                                        series='Train_accuracy', value=train_accuracy, iteration=self.epoch)
            self.val_net()

    def val_net(self):
        self.net.eval()
        correct, total = 0, 0
        running_loss = 0
        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                inputs, labels = [i.to(self.device) for i in data]
                outputs = self.net(inputs)
                pred = torch.argmax(outputs, 1)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        accuracy = correct/total
        val_loss = running_loss / (i + 1)
        model_saved = False
        if accuracy > self.best_metric:
            torch.save(self.net.state_dict(), os.path.join(self.exp_folder, f"best_model.pth"))
            model_saved = True
        print(f"Val_loss: {running_loss}, val_accuracy: {round(accuracy, 4)}, Model saved: {model_saved}")
        self.logger.report_scalar(title='Train vs Val loss',
                                        series='Val_loss', value=val_loss, iteration=self.epoch)
        self.logger.report_scalar(title='Train vs Val accuracy',
                                    series='Val_accuracy', value=accuracy, iteration=self.epoch)

    def create_exp_folder(self):
        exp_folder = os.path.join(self.config["DIR_PATH"], "exp_folder")
        if not os.path.exists(exp_folder):
            IO.create_dir(exp_folder)
        now = datetime.datetime.now()
        exp_id = "Exp_" + str(now.date()) + '-' + ''.join(
            [str(now.hour).zfill(2), str(now.minute).zfill(2), str(now.second).zfill(2)])
        self.exp_folder = os.path.join(exp_folder, exp_id)
        self.config["exp_folder"] = self.exp_folder
        IO.create_dir(self.exp_folder)

        
    def run(self):
        self.create_exp_folder()
        self.task.set_parameters(self.config)
        # self.config = self.task.connect_configuration(self.config)
        self.load_data()
        self.upload_net()
        self.train_net()

        # self.load_data()
        # dataiter = iter(self.train_loader)
        # images, labels = dataiter.next()
        # torch.max(images)