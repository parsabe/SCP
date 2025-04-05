import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

optimizers = ["Adam", "SGD"]
PLOT_PATH = "plots"
os.makedirs(PLOT_PATH, exist_ok=True)

for opt in optimizers:
    for fold in range(3):  # You have fold_0, fold_1, fold_2
        file = f"training_history_{opt}_fold_{fold}.npy"
        if not os.path.exists(file):
            print(f"File not found: {file}")
            continue

        history = np.load(file, allow_pickle=True).item()
        acc = [a * 100 for a in history["train_acc"]]  # Scale to percentage
        loss = history["train_loss"]

        preds = np.load(f"all_preds_{opt}.npy")
        labels = np.load(f"all_labels_{opt}.npy")
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(10, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {opt}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(f"plots/confusion_matrix_{opt}.png")
        plt.close()
        # 📈 Training Accuracy
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=range(len(acc)), y=acc)
        plt.title(f"Training Accuracy - {opt}")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_PATH, f"training_accuracy_{opt}_fold_{fold}.png"))
        plt.close()

        # 📉 Training Loss
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=range(len(loss)), y=loss)
        plt.title(f"Training Loss - {opt}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_PATH, f"training_loss_{opt}_fold_{fold}.png"))
        plt.close()
