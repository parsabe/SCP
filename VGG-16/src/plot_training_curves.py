import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PLOT_PATH = "plots"
os.makedirs(PLOT_PATH, exist_ok=True)

optimizers = ["Adam", "SGD"]
folds = [0, 1, 2]

for opt in optimizers:
    all_acc = []
    all_loss = []

    for fold in folds:
        file = f"training_history_{opt}_fold_{fold}.npy"
        if not os.path.exists(file):
            print(f"Missing file: {file}")
            continue

        history = np.load(file, allow_pickle=True).item()
        acc = history["train_acc"] 
        loss = history["train_loss"]

        all_acc.append(acc)
        all_loss.append(loss)

        # Training Accuracy - per fold
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=range(len(acc)), y=acc)
        plt.title(f"Training Accuracy - {opt} - Fold{fold}")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_PATH, f"training_accuracy_{opt}_fold_{fold}.png"))
        plt.close()

        # Training Loss - per fold
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=range(len(loss)), y=loss)
        plt.title(f"Training Loss - {opt} - Fold{fold}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_PATH, f"training_loss_{opt}_fold_{fold}.png"))
        plt.close()

    if all_acc and all_loss:
        # Truncate to minimum length across folds
        min_len = min(len(a) for a in all_acc)
        all_acc = [a[:min_len] for a in all_acc]
        all_loss = [l[:min_len] for l in all_loss]

        avg_acc = np.mean(all_acc, axis=0)
        avg_loss = np.mean(all_loss, axis=0)

        # Training Accuracy - averaged
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=range(len(avg_acc)), y=avg_acc)
        plt.title(f"Training Accuracy Average - {opt} - Fold{fold}")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_PATH, f"training_accuracy_{opt}_avg.png"))
        plt.close()

        # Training Loss - averaged
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=range(len(avg_loss)), y=avg_loss)
        plt.title(f"Training Loss Average - {opt} - Fold{fold}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_PATH, f"training_loss_{opt}_avg.png"))
        plt.close()

