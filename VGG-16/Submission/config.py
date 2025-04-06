import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import load_dataset
import random
import os


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 10
K_FOLDS = 3
PATIENCE = 7
NUM_EPOCHS = 50

BASE_PATH = "/home/username/..."
PLOT_PATH = os.path.join(BASE_PATH, "plots")
EVAL_PATH = os.path.join(BASE_PATH, "evaluation")
PARAMS_PATH = os.path.join(BASE_PATH, "best_params")


