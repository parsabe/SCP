# Combined Evaluation and Plotting Script (Full Version)

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    f1_score
)
from config import torch, DEVICE, NUM_CLASSES, load_dataset, PLOT_PATH, EVAL_PATH
from model import SimplifiedVGG

os.makedirs(PLOT_PATH, exist_ok=True)
os.makedirs(EVAL_PATH, exist_ok=True)


def evaluate_model(model_name, visualize=True):
    model = SimplifiedVGG().to(DEVICE)
    model.load_state_dict(torch.load(f"best_model_{model_name}.pth"))
    model.eval()

    _, test_loader = load_dataset.load_dataset(batch_size=32, path="/scratch/pb70gygu")

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE).long()
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, digits=4)

    with open(os.path.join(EVAL_PATH, f"classification_report_model_{model_name}.txt"), "w") as f:
        f.write(f"Test Accuracy: {acc * 100:.2f}%\n")
        f.write(report)

    if visualize:
        visualize_results(all_labels, all_preds, all_probs, test_loader, model_name)

    return acc


def visualize_results(all_labels, all_preds, all_probs, test_loader, model_name):
    classes = list(range(NUM_CLASSES))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, f"confusion_matrix_{model_name}.png"))
    plt.close()

    # Misclassified Samples
    misclassified_indices = [i for i, (label, pred) in enumerate(zip(all_labels, all_preds)) if label != pred]
    plot_samples_by_index(test_loader, misclassified_indices[:5], model_name, "misclassified")

    # Lowest Confidence Samples
    confidences = np.max(all_probs, axis=1)
    lowest_indices = np.argsort(confidences)[:5]
    plot_samples_by_index(test_loader, lowest_indices, model_name, "lowest_confidence")

    # Highest Error Samples
    error_distances = np.abs(np.array(all_preds) - np.array(all_labels))
    highest_error_indices = np.argsort(error_distances)[-5:]
    plot_samples_by_index(test_loader, highest_error_indices, model_name, "highest_error")


def plot_samples_by_index(test_loader, indices, model_name, tag):
    plt.figure(figsize=(15, 6))
    shown = 0
    flat_loader = [(img, lbl) for batch in test_loader for img, lbl in zip(*batch)]
    for idx in indices:
        image, label = flat_loader[idx]
        plt.subplot(1, len(indices), shown + 1)
        plt.imshow(np.transpose(image.cpu().numpy(), (1, 2, 0)))
        plt.title(f"True: {label}")
        plt.axis("off")
        shown += 1
    plt.suptitle(f"{tag.replace('_', ' ').title()} Samples - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, f"{tag}_samples_{model_name}.png"))
    plt.close()


def plot_training_curves():
    optimizers = ["Adam", "SGD"]
    folds = [0, 1, 2]

    for opt in optimizers:
        all_acc = []
        all_loss = []

        for fold in folds:
            file = f"training_history_{opt}_fold_{fold}.npy"
            if not os.path.exists(file):
                continue
            history = np.load(file, allow_pickle=True).item()
            acc = history["train_acc"]
            loss = history["train_loss"]

            all_acc.append(acc)
            all_loss.append(loss)

            plt.figure(figsize=(10, 6))
            sns.lineplot(x=range(len(acc)), y=acc)
            plt.title(f"Training Accuracy - {opt} - Fold{fold}")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_PATH, f"training_accuracy_{opt}_fold_{fold}.png"))
            plt.close()

            plt.figure(figsize=(10, 6))
            sns.lineplot(x=range(len(loss)), y=loss)
            plt.title(f"Training Loss - {opt} - Fold{fold}")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_PATH, f"training_loss_{opt}_fold_{fold}.png"))
            plt.close()

        if all_acc and all_loss:
            min_len = min(len(a) for a in all_acc)
            all_acc = [a[:min_len] for a in all_acc]
            all_loss = [l[:min_len] for l in all_loss]

            avg_acc = np.mean(all_acc, axis=0)
            avg_loss = np.mean(all_loss, axis=0)

            plt.figure(figsize=(10, 6))
            sns.lineplot(x=range(len(avg_acc)), y=avg_acc)
            plt.title(f"Training Accuracy Average - {opt}")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_PATH, f"training_accuracy_{opt}_avg.png"))
            plt.close()

            plt.figure(figsize=(10, 6))
            sns.lineplot(x=range(len(avg_loss)), y=avg_loss)
            plt.title(f"Training Loss Average - {opt}")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_PATH, f"training_loss_{opt}_avg.png"))
            plt.close()


def main():
    results = {}
    for opt in ["Adam", "SGD"]:
        acc = evaluate_model(opt)
        results[opt] = acc * 100

    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(results.keys()), y=list(results.values()), palette="coolwarm")
    plt.title("Final Test Accuracy Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "optimizer_comparison.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(results.keys()), y=list(results.values()), palette="viridis")
    for i, v in enumerate(results.values()):
        plt.text(i, v + 1, f"{v:.2f}%", ha="center")
    plt.title("Evaluation Results Chart")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "evaluation_results_chart.png"))
    plt.close()

    plot_training_curves()


if __name__ == "__main__":
    main()
