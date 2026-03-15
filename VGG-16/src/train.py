import torch
import torch.nn as nn
import torchvision.models as models
from config import np, optim, DEVICE, NUM_CLASSES, NUM_EPOCHS, K_FOLDS, PATIENCE, KFold, DataLoader, Subset, load_dataset
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
        
        # 1. STRICT GPU CHECK
        assert torch.cuda.is_available(), "CRITICAL: CUDA is not available. This script strictly requires a GPU."
        self.device = torch.device("cuda") 
        
        self.num_classes = NUM_CLASSES
        self.num_epochs = NUM_EPOCHS
        self.k_folds = K_FOLDS
        self.patience = PATIENCE
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        
        # Data loader initialization remains the same
        self.train_loader, _ = load_dataset.load_dataset(batch_size=self.batch_size, path="/scratch/pb70gygu")
        self.kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)

    def get_optimizer(self, model):
        # ... [Keep your existing get_optimizer logic exactly as it is] ...
        pass 

    def train(self, return_best_val_acc=False):
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(self.kf.split(range(len(self.train_loader.dataset)))):
            print(f"\nFold {fold+1}/{self.k_folds}")
            
            best_val_loss = np.inf
            epochs_no_improve = 0
            history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

            train_subset = Subset(self.train_loader.dataset, train_idx)
            val_subset = Subset(self.train_loader.dataset, val_idx)
            
            # Added pin_memory=True to speed up CPU-to-GPU data transfers
            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)

            # 2 & 3. LOAD RESNET AND PARALLELIZE
            # Using resnet18 here as a standard, can be changed to resnet50
            model = models.resnet18(weights=None) 
            
            # Adjust the final layer for your specific dataset
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.num_classes)
            
            # Wrap model in DataParallel if multiple GPUs exist
            if torch.cuda.device_count() > 1:
                print(f"Parallelizing across {torch.cuda.device_count()} GPUs!")
                model = nn.DataParallel(model)
                
            model = model.to(self.device)

            criterion = nn.CrossEntropyLoss(label_smoothing=0.01)
            optimizer, scheduler = self.get_optimizer(model)

            for epoch in range(self.num_epochs):
                model.train()
                running_loss, correct_train, total_train = 0.0, 0, 0

                for images, labels in train_loader:
                    # Move data to GPU non-blockingly for faster execution
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True).long()
                    
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

                # ... [End of your val_loader loop] ...
                val_accuracy = 100 * correct_val / total_val
                print(f"Epoch [{epoch+1}/{self.num_epochs}], Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {val_accuracy:.2f}%")

                epoch_train_loss = running_loss / len(train_loader)
                epoch_train_acc = 100 * correct_train / total_train
                epoch_val_loss = val_loss / len(val_loader)

                # Checkpoint & Early Stopping Logic
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    best_val_acc_for_fold = val_accuracy # FIX: Track the best accuracy!
                    epochs_no_improve = 0
                    
                    # FIX: Handle nn.DataParallel wrapper when saving weights
                    if isinstance(model, torch.nn.DataParallel):
                        state_dict_to_save = model.module.state_dict()
                    else:
                        state_dict_to_save = model.state_dict()
                        
                    torch.save(state_dict_to_save, f"best_model_{self.optimizer_type}.pth")
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

                # Step the scheduler (Note: Make sure OneCycleLR steps per batch, not epoch!)
                if scheduler:
                    scheduler.step()

            # End of Epoch Loop
            
            # FIX: Append the *best* accuracy achieved during the fold, not the last/degraded one
            fold_results.append(best_val_acc_for_fold) 
            np.save(f"training_history_{self.optimizer_type}_fold_{fold}.npy", history)
            print(f"Training history saved for fold {fold}.")

        # End of Fold Loop
        avg_val_acc = np.mean(fold_results)
        print(f"\nAverage Validation Accuracy ({self.optimizer_type}): {avg_val_acc:.2f}%")
        print(f"Estimated Test Error: {100 - avg_val_acc:.2f}%")

        if return_best_val_acc:
            return max(fold_results)