import os
import sys

PROJECT_ROOT = 'D:/multitask_prediction'
sys.path.insert(0, PROJECT_ROOT)

from app.train import train_model

if __name__ == "__main__":
    train_model()