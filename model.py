from config import torch, nn, NUM_CLASSES, optim
import numpy as np
import random
from sklearn.model_selection import ParameterGrid

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class SimplifiedVGG(nn.Module):
    def __init__(self, dropout_rate=0.4):
        super(SimplifiedVGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(), nn.Dropout(dropout_rate),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.LeakyReLU(), nn.Dropout(dropout_rate),
            nn.Linear(512, NUM_CLASSES)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

# Hyperparameter tuning
param_grid = {
    "learning_rate": [0.01, 0.001, 0.0001],
    "batch_size": [16, 32, 64],
    "dropout_rate": [0.3, 0.4, 0.5],
    "optimizer": ["Adam", "SGD", "RMSprop"]
}

def get_optimizer(model, optimizer_name, learning_rate):
    if optimizer_name == "Adam":
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "SGD":
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name == "RMSprop":
        return optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Unsupported optimizer type")

# Generate all combinations of hyperparameters
hyperparameter_combinations = list(ParameterGrid(param_grid))
