import os 
import sys
import json

DIR_PATH = os.path.abspath(".")
sys.path.append(os.path.join(DIR_PATH, "packages"))
from Cifar_classification import CifarTrain

def read_json(fn):
    with open(fn, "r") as fin:
        try:
            return json.load(fin)
        except Exception as e:
            raise IOError(f"Cannot read the file, error: {e}")

