from config import torch, sns, plt, os, np, confusion_matrix, DEVICE, NUM_CLASSES, load_dataset, PLOT_PATH, EVAL_PATH
from model import SimplifiedVGG
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, precision_recall_curve, f1_score
from collections import Counter
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

def evaluate_model(model_name, visualize=True):
    model = SimplifiedVGG().to(DEVICE)
    model.load_state_dict(torch.load(f"best_model_{model_name}.pth"))
    model.eval()

    _, test_loader = load_dataset.load_dataset(batch_size=32, path="/scratch/username")

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
    print(f"\nEvaluation for Optimizer: {model_name}")
    print(f"Test Accuracy: {acc * 100:.2f}%")
    report = classification_report(all_labels, all_preds, digits=4)
    print("\nClassification Report:")
    print(report)

    os.makedirs(EVAL_PATH, exist_ok=True)
    with open(os.path.join(EVAL_PATH, f"classification_report_model_{model_name}.txt"), "w") as f:
        f.write(f"Test Accuracy: {acc * 100:.2f}%\n")
        f.write(report)

    if visualize:
        visualize_results(all_labels, all_preds, all_probs, test_loader, model_name, acc)

    return acc


def visualize_results(all_labels, all_preds, all_probs, test_loader, model_name, acc):
    os.makedirs(PLOT_PATH, exist_ok=True)
    classes = list(range(NUM_CLASSES))

    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, f"confusion_matrix_{model_name}.png"))
    plt.close()




if __name__ == "__main__":
    results = {}

    for opt in ["Adam", "SGD"]:
        acc = evaluate_model(model_name=opt, visualize=True)
        results[opt] = acc * 100

    print("\nEvaluation Results:")
    for opt, acc in results.items():
        print(f"{opt}: {acc:.2f}% Test Accuracy")

    best = max(results, key=results.get)
    print(f"\nBest Optimizer: {best} with Accuracy: {results[best]:.2f}%")
 
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(results.keys()), y=list(results.values()), palette="viridis")
    plt.title("Test Accuracy Comparison")
    for i, v in enumerate(results.values()):
        plt.text(i, v + 1, f"{v:.2f}%", ha="center", fontsize=12)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "evaluation_results_chart.png"))
    plt.close()


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
