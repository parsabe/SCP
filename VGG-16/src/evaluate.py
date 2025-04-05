from config import torch, sns, plt, os, np, confusion_matrix, DEVICE, NUM_CLASSES, load_dataset, PLOT_PATH, EVAL_PATH
from model import SimplifiedVGG
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, precision_recall_curve, f1_score
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.calibration import calibration_curve
import json

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

   
    class_correct = Counter()
    class_total = Counter()
    for label, pred in zip(all_labels, all_preds):
        if label == pred:
            class_correct[label] += 1
        class_total[label] += 1
    accuracy = [100 * class_correct[c] / class_total[c] if class_total[c] > 0 else 0 for c in classes]

    plt.figure(figsize=(12, 6))
    sns.barplot(x=classes, y=accuracy, palette="magma")
    plt.title(f"Per-Class Accuracy - {model_name}")
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, f"per_class_accuracy_{model_name}.png"))
    plt.close()

   
    errors = [label for label, pred in zip(all_labels, all_preds) if label != pred]
    error_counts = Counter(errors)
    counts = [error_counts.get(c, 0) for c in classes]

    plt.figure(figsize=(12, 6))
    sns.barplot(x=classes, y=counts, palette="rocket")
    plt.title(f"Error Distribution - {model_name}")
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, f"error_distribution_{model_name}.png"))
    plt.close()

    
    plt.figure(figsize=(10, 8))
    for c in classes:
        fpr, tpr, _ = roc_curve([1 if l == c else 0 for l in all_labels], [p[c] for p in all_probs])
        sns.lineplot(x=fpr, y=tpr, label=f"Class {c} (AUC={auc(fpr, tpr):.2f})")
    plt.title(f"ROC Curve - {model_name}")
    sns.despine()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, f"roc_curve_{model_name}.png"))
    plt.close()

   
    plt.figure(figsize=(10, 8))
    for c in classes:
        precision, recall, _ = precision_recall_curve([1 if l == c else 0 for l in all_labels], [p[c] for p in all_probs])
        sns.lineplot(x=recall, y=precision, label=f"Class {c}")
    plt.title(f"Precision-Recall Curve - {model_name}")
    sns.despine()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, f"precision_recall_curve_{model_name}.png"))
    plt.close()

   
    f1_scores = f1_score(all_labels, all_preds, average=None)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=classes, y=f1_scores, palette="viridis")
    plt.title(f"F1 Score per Class - {model_name}")
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, f"f1_score_{model_name}.png"))
    plt.close()

    
    plot_misclassified_samples(test_loader, all_labels, all_preds, model_name, max_samples=5)

   
    confidences = np.max(all_probs, axis=1)
    lowest_indices = np.argsort(confidences)[:5]
    plot_samples_by_index(test_loader, lowest_indices, model_name, "lowest_confidence")

    
    error_distances = np.abs(np.array(all_preds) - np.array(all_labels))
    highest_error_indices = np.argsort(error_distances)[-5:]
    plot_samples_by_index(test_loader, highest_error_indices, model_name, "highest_error")

   
    plt.figure(figsize=(10, 8))
    for c in classes:
        y_true_bin = [1 if l == c else 0 for l in all_labels]
        y_prob = [p[c] for p in all_probs]
        prob_true, prob_pred = calibration_curve(y_true_bin, y_prob, n_bins=10, strategy="uniform")
        sns.lineplot(x=prob_pred, y=prob_true, marker="o", label=f"Class {c}")
    sns.lineplot(x=[0, 1], y=[0, 1], linestyle="--", color="black")
    plt.title(f"Calibration Curve - {model_name}")
    sns.despine()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, f"calibration_curve_{model_name}.png"))
    plt.close()


    tsne = TSNE(n_components=2, random_state=42)
    embeddings = tsne.fit_transform(np.array(all_probs))
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], hue=all_labels, palette="tab10", legend="full")
    plt.title(f"t-SNE Embedding - {model_name}")
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, f"tsne_{model_name}.png"))
    plt.close()

 
    plt.figure(figsize=(10, 6))
    sns.histplot(all_preds, bins=NUM_CLASSES)
    plt.title(f"Prediction Histogram - {model_name}")
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, f"prediction_histogram_{model_name}.png"))
    plt.close()

    
    tuning_results = {
        "Adam": {"best": 67.33, "worst": 45.20},
        "SGD": {"best": 47.47, "worst": 35.12}
    }
    plt.figure(figsize=(10, 6))
    for opt, scores in tuning_results.items():
        sns.barplot(x=["Best", "Worst"], y=[scores["best"], scores["worst"]], label=opt)
    plt.title("Hyperparameter Tuning Results")
    plt.legend()
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "tuning_results.png"))
    plt.close()


def plot_misclassified_samples(test_loader, all_labels, all_preds, model_name, max_samples=5):
    misclassified_indices = [i for i, (label, pred) in enumerate(zip(all_labels, all_preds)) if label != pred]
    if not misclassified_indices:
        print("No misclassified samples!")
        return
    plot_samples_by_index(test_loader, misclassified_indices[:max_samples], model_name, "misclassified")


def plot_samples_by_index(test_loader, indices, model_name, tag):
    plt.figure(figsize=(15, 6))
    shown = 0
    for batch in test_loader:
        images, labels = batch
        for i in range(len(images)):
            if shown >= len(indices):
                break
            idx = shown
            image = images[i]
            label = labels[i]
            plt.subplot(1, len(indices), shown + 1)
            plt.imshow(np.transpose(image.cpu().numpy(), (1, 2, 0)))
            plt.title(f"True: {label}")
            plt.axis("off")
            shown += 1
        if shown >= len(indices):
            break
    plt.suptitle(f"{tag.capitalize()} Samples - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, f"{tag}_samples_{model_name}.png"))
    plt.close()


if __name__ == "__main__":
    results = {}

    for opt in ["Adam", "SGD"]:
        acc = evaluate_model(model_name=opt, visualize=True)
        results[opt] = acc * 100

    print("\n📊 Evaluation Results:")
    for opt, acc in results.items():
        print(f"{opt}: {acc:.2f}% Test Accuracy")

    best = max(results, key=results.get)
    print(f"\n🏆 Best Optimizer: {best} with Accuracy: {results[best]:.2f}%")

    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(results.keys()), y=list(results.values()), palette="coolwarm")
    plt.title("Final Test Accuracy Comparison")
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "optimizer_comparison.png"))
    plt.close()

   
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(results.keys()), y=list(results.values()), palette="viridis")
    plt.title("📊 Evaluation Results")
    for i, v in enumerate(results.values()):
        plt.text(i, v + 1, f"{v:.2f}%", ha="center", fontsize=12)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "evaluation_results_chart.png"))
    plt.close()
