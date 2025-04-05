import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


optimizers = ["Adam", "SGD"]
PLOT_PATH = "plots"
os.makedirs(PLOT_PATH, exist_ok=True)

for opt in optimizers:
    for fold in range(3):  
        file = f"training_history_{opt}_fold_{fold}.npy"
        if not os.path.exists(file):
            print(f"File not found: {file}")
            continue

        history = np.load(file, allow_pickle=True).item()
        acc = [a * 100 for a in history["train_acc"]]  # Scale to percentage
        loss = history["train_loss"]

    

        plt.figure(figsize=(10, 6))
        sns.lineplot(x=range(len(acc)), y=acc)
        plt.title(f"Training Accuracy - {opt}")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_PATH, f"training_accuracy_{opt}_fold_{fold}.png"))
        plt.close()

     
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=range(len(loss)), y=loss)
        plt.title(f"Training Loss - {opt}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_PATH, f"training_loss_{opt}_fold_{fold}.png"))
        plt.close()
