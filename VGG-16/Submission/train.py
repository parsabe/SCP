from config import torch, np, optim, DEVICE, NUM_CLASSES, NUM_EPOCHS, K_FOLDS, PATIENCE, KFold, DataLoader, Subset, load_dataset
from model import SimplifiedVGG
import torchvision.transforms as transforms
import random
import os


SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)


def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class Trainer:
    def __init__(self, learning_rate=0.001, batch_size=32, optimizer_type="Adam"):
        self.batch_size = batch_size
        self.device = DEVICE
        self.num_classes = NUM_CLASSES
        self.num_epochs = NUM_EPOCHS
        self.k_folds = K_FOLDS
        self.patience = PATIENCE
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.train_loader, _ = load_dataset.load_dataset(batch_size=self.batch_size, path="/scratch/username")
        self.kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=SEED)

    def get_optimizer(self, model):
        if self.optimizer_type == "Adam":
            optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        elif self.optimizer_type == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4, nesterov=True)
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                steps_per_epoch=len(self.train_loader),
                epochs=self.num_epochs
            )
        else:
            raise ValueError("Unsupported optimizer type")
        return optimizer, scheduler

    def train(self, return_best_val_acc=False):
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(self.kf.split(range(len(self.train_loader.dataset)))):
            print(f"\nFold {fold+1}/{self.k_folds}")
            
            best_val_loss = np.inf
            epochs_no_improve = 0
            history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

            train_subset = Subset(self.train_loader.dataset, train_idx)
            val_subset = Subset(self.train_loader.dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False, num_workers=4)

            model = SimplifiedVGG().to(self.device)
            criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.01)
            optimizer, scheduler = self.get_optimizer(model)

            for epoch in range(self.num_epochs):
                model.train()
                running_loss, correct_train, total_train = 0.0, 0, 0

                for images, labels in train_loader:
                    images, labels = images.to(self.device), labels.to(self.device).long()
                    optimizer.zero_grad()
                    images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.2)
                    outputs = model(images)
                    loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    correct_train += (predicted == labels).sum().item()
                    total_train += labels.size(0)

               
                model.eval()
                correct_val, total_val, val_loss = 0, 0, 0.0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(self.device), labels.to(self.device).long()
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        correct_val += (predicted == labels).sum().item()
                        total_val += labels.size(0)

                val_accuracy = 100 * correct_val / total_val
                print(f"Epoch [{epoch+1}/{self.num_epochs}], Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {val_accuracy:.2f}%")

                epoch_train_loss = running_loss / len(train_loader)
                epoch_train_acc = 100 * correct_train / total_train
                epoch_val_loss = val_loss / len(val_loader)

                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    epochs_no_improve = 0
                    torch.save(model.state_dict(), f"best_model_{self.optimizer_type}.pth")
                    print(f"Val Loss Improved. Model saved at epoch {epoch+1}")
                else:
                    epochs_no_improve += 1
                    print(f"No improvement for {epochs_no_improve} epoch(s)")

                if epochs_no_improve >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch+1} due to no improvement for {self.patience} epochs.")
                    break

                history["train_loss"].append(epoch_train_loss)
                history["val_loss"].append(epoch_val_loss)
                history["train_acc"].append(epoch_train_acc)
                history["val_acc"].append(val_accuracy)

                if scheduler:
                    scheduler.step()

            fold_results.append(val_accuracy)
            np.save(f"training_history_{self.optimizer_type}_fold_{fold}.npy", history)
            print(f"Training history saved for fold {fold}.")

        avg_val_acc = np.mean(fold_results)
        print(f"\n Average Validation Accuracy ({self.optimizer_type}): {avg_val_acc:.2f}%")
        print(f"Estimated Test Error: {100 - avg_val_acc:.2f}%")

        if return_best_val_acc:
            return max(fold_results)
