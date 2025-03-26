from config import torch, optim, DEVICE, NUM_CLASSES, NUM_EPOCHS, BATCH_SIZE, K_FOLDS, KFold, DataLoader, Subset, load_dataset
from model import SimplifiedVGG
import torchvision.transforms as transforms
import random

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)


# Data augmentation transformations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(56, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class Trainer:
    def __init__(self, learning_rate=0.001, optimizer_type="Adam"):
        self.device = DEVICE
        self.num_classes = NUM_CLASSES
        self.num_epochs = NUM_EPOCHS
        self.batch_size = BATCH_SIZE
        self.k_folds = K_FOLDS
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.train_loader, _ = load_dataset.load_dataset(batch_size=self.batch_size, path="./data")
        self.kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=SEED)

    def get_optimizer(self, model):
        if self.optimizer_type == "Adam":
            return optim.Adam(model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == "SGD":
            return optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-4, nesterov=True)
        elif self.optimizer_type == "RMSprop":
            return optim.RMSprop(model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError("Unsupported optimizer type")

    def train(self):
        for fold, (train_idx, val_idx) in enumerate(self.kf.split(range(len(self.train_loader.dataset)))):
            print(f"Fold {fold+1}/{self.k_folds}")
            train_subset = Subset(self.train_loader.dataset, train_idx)
            val_subset = Subset(self.train_loader.dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False, num_workers=4)
            
            model = SimplifiedVGG().to(self.device)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = self.get_optimizer(model)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs)
            
            for epoch in range(self.num_epochs):
                model.train()
                running_loss, correct_train, total_train = 0.0, 0, 0
                for images, labels in train_loader:
                    images, labels = images.to(self.device), labels.to(self.device).long()
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    correct_train += (predicted == labels).sum().item()
                    total_train += labels.size(0)
                
                scheduler.step()
            
            torch.save(model.state_dict(), "trained_model.pth")
            print("Model saved as 'trained_model.pth'.")

if __name__ == "__main__":
    trainer = Trainer(learning_rate=0.01, optimizer_type="SGD")
    trainer.train()