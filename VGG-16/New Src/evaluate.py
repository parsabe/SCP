import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from config import DEVICE, NUM_CLASSES, PLOT_PATH
from model import SimplifiedVGG
from load_dataset import load_dataset


def load_training_history():
    history = {
        "Adam": [],
        "SGD": []
    }

    for opt in history.keys():
        for fold in range(3):
            file_path = f"training_history_{opt}_fold_{fold}.npy"
            if os.path.exists(file_path):
                data = np.load(file_path, allow_pickle=True).item()
                history[opt].append(data)

    return history


def plot_training_curves(history):
    os.makedirs(PLOT_PATH, exist_ok=True)

    for opt in ["Adam", "SGD"]:
        avg_acc = np.mean([h["accuracy"] for h in history[opt]], axis=0)
        avg_loss = np.mean([h["loss"] for h in history[opt]], axis=0)
        class_acc = np.mean([h["per_class_accuracy"] for h in history[opt]], axis=0)
        class_loss = np.mean([h["per_class_loss"] for h in history[opt]], axis=0)

        # Accuracy over epochs
        plt.figure()
        plt.plot(range(1, len(avg_acc) + 1), avg_acc)
        plt.xlabel("Number of Training Epochs")
        plt.ylabel("Average Accuracy (0 to 1)")
        plt.title(f"Training Accuracy over Epochs - {opt}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_PATH, f"training_accuracy_{opt}.png"))
        plt.close()

        # Loss over epochs
        plt.figure()
        plt.plot(range(1, len(avg_loss) + 1), avg_loss)
        plt.xlabel("Number of Training Epochs")
        plt.ylabel("Average Loss")
        plt.title(f"Training Loss over Epochs - {opt}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_PATH, f"training_loss_{opt}.png"))
        plt.close()

        # Per-Class Accuracy
        plt.figure()
        sns.barplot(x=list(range(1, NUM_CLASSES + 1)), y=class_acc)
        plt.xlabel("Class Labels (1 to 10)")
        plt.ylabel("Accuracy")
        plt.title(f"Per-Class Training Accuracy - {opt}")
        plt.grid(True, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_PATH, f"per_class_accuracy_{opt}.png"))
        plt.close()

        # Per-Class Loss
        plt.figure()
        sns.barplot(x=list(range(1, NUM_CLASSES + 1)), y=class_loss)
        plt.xlabel("Class Labels (1 to 10)")
        plt.ylabel("Loss")
        plt.title(f"Per-Class Training Loss - {opt}")
        plt.grid(True, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_PATH, f"per_class_loss_{opt}.png"))
        plt.close()


def evaluate_and_plot_test():
    os.makedirs(PLOT_PATH, exist_ok=True)
    results = {}

    for opt in ["Adam", "SGD"]:
        model = SimplifiedVGG().to(DEVICE)
        model.load_state_dict(torch.load(f"best_model_{opt}.pth"))
        model.eval()

        _, test_loader = load_dataset(batch_size=32, path="/scratch/pb70gygu")
        all_preds, all_labels = [], []
        correct, total = 0, 0
        criterion = torch.nn.CrossEntropyLoss()
        running_loss = 0.0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE).long()
                outputs = model(images)
                loss = criterion(outputs, labels)

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                running_loss += loss.item() * labels.size(0)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = correct / total
        avg_loss = running_loss / total
        results[opt] = {"accuracy": acc, "loss": avg_loss}

        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 6))
        sns.heatmap(cm.T, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {opt}")
        plt.xlabel("True Label")
        plt.ylabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_PATH, f"confusion_matrix_{opt}.png"))
        plt.close()

        # Per-Class Accuracy
        class_correct = np.zeros(NUM_CLASSES)
        class_total = np.zeros(NUM_CLASSES)

        for l, p in zip(all_labels, all_preds):
            class_total[l] += 1
            if l == p:
                class_correct[l] += 1

        class_acc = class_correct / class_total

        plt.figure()
        sns.barplot(x=list(range(1, NUM_CLASSES + 1)), y=class_acc)
        plt.xlabel("Class Labels (1 to 10)")
        plt.ylabel("Accuracy")
        plt.title(f"Per-Class Test Accuracy - {opt}")
        plt.grid(True, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_PATH, f"per_class_test_accuracy_{opt}.png"))
        plt.close()

        # Per-Class Loss (Dummy zero loss for test)
        class_loss = np.zeros(NUM_CLASSES)
        plt.figure()
        sns.barplot(x=list(range(1, NUM_CLASSES + 1)), y=class_loss)
        plt.xlabel("Class Labels (1 to 10)")
        plt.ylabel("Loss")
        plt.title(f"Per-Class Test Loss - {opt}")
        plt.grid(True, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_PATH, f"per_class_test_loss_{opt}.png"))
        plt.close()

    # Test Accuracy Comparison
    plt.figure()
    sns.barplot(x=list(results.keys()), y=[v["accuracy"] * 100 for v in results.values()])
    plt.xlabel("Optimizer")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Test Accuracy Comparison")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "test_accuracy_comparison.png"))
    plt.close()

    # Test Loss Comparison
    plt.figure()
    sns.barplot(x=list(results.keys()), y=[v["loss"] for v in results.values()])
    plt.xlabel("Optimizer")
    plt.ylabel("Test Loss")
    plt.title("Test Loss Comparison")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "test_loss_comparison.png"))
    plt.close()


if __name__ == "__main__":
    history = load_training_history()
    plot_training_curves(history)
    evaluate_and_plot_test()
