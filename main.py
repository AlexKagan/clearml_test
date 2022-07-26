import os 
import sys
import json

DIR_PATH = os.path.abspath(".")
sys.path.append(os.path.join(DIR_PATH, "packages"))
from Mnist_classification import MnistTrain
from utils import IO

if __name__ == '__main__':
    # config = IO.read_json(os.path.join(DIR_PATH, "config", "mnist_config.json"))
    # config["DIR_PATH"] = DIR_PATH
    # print(f"Config: {config}")
    MnistTrain(DIR_PATH)