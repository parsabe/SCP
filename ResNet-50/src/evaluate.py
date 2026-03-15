import torch
import torch.nn as nn
import torchvision.models as models
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc, precision_recall_curve, f1_score
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.calibration import calibration_curve

# Assuming these are correctly imported from your config.py
from config import DEVICE, NUM_CLASSES, load_dataset, PLOT_PATH, EVAL_PATH

def evaluate_model(model_name, visualize=True):
    print(f"\n--- Loading and Evaluating Model: {model_name} ---")
    
    # 1. FIX: Rebuild the exact ResNet-18 architecture used in training
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    
    # 2. FIX: Robust state_dict loading
    model_path = f"best_model_{model_name}.pth" # Make sure this matches how you saved it! (e.g., you added _fold0 in the train script)
    if not os.path.exists(model_path):
        print(f"ERROR: Could not find {model_path}. Skipping...")
        return 0.0
        
    state_dict = torch.load(model_path, map_location=DEVICE)
    
    # Strip 'module.' prefix if it accidentally got saved with it
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    
    # Use multiple GPUs for evaluation if available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    model = model.to(DEVICE)
    model.eval()

    _, test_loader = load_dataset.load_dataset(batch_size=32, path=r"C:\Users\parsa\Desktop\code\SCP\ResNet-50\data")

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            # Use non_blocking for speed
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True).long()
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\nEvaluation for Optimizer: {model_name}")
    print(f"Test Accuracy: {acc * 100:.2f}%")
    
    # Wrap in try-except in case a class is never predicted
    try:
        report = classification_report(all_labels, all_preds, digits=4)
        print("\nClassification Report:")
        print(report)

        os.makedirs(EVAL_PATH, exist_ok=True)
        with open(os.path.join(EVAL_PATH, f"classification_report_model_{model_name}.txt"), "w") as f:
            f.write(f"Test Accuracy: {acc * 100:.2f}%\n")
            f.write(report)
    except Exception as e:
        print(f"Could not generate classification report: {e}")

    if visualize:
        # Pass the actual dataset object so we can extract specific images by index later
        visualize_results(all_labels, all_preds, all_probs, test_loader.dataset, model_name, acc)

    return acc

def visualize_results(all_labels, all_preds, all_probs, test_dataset, model_name, acc):
    os.makedirs(PLOT_PATH, exist_ok=True)
    classes = list(range(NUM_CLASSES))

    # --- Metrics Plots (Confusion Matrix, Per-class, Error Dist, ROC, PR, F1, Calib, TSNE, Hist) ---
    # [Your existing plotting code here is excellent and structurally sound. 
    #  To save space, I have omitted repeating it, but you should keep it exactly as it is!]
    # ... 

    # --- FIX: Pass test_dataset instead of test_loader to these functions ---
    plot_misclassified_samples(test_dataset, all_labels, all_preds, model_name, max_samples=5)

    confidences = np.max(all_probs, axis=1)
    lowest_indices = np.argsort(confidences)[:5]
    plot_samples_by_index(test_dataset, lowest_indices, model_name, "lowest_confidence")

    error_distances = np.abs(np.array(all_preds) - np.array(all_labels))
    highest_error_indices = np.argsort(error_distances)[-5:]
    plot_samples_by_index(test_dataset, highest_error_indices, model_name, "highest_error")
    # ... [Keep your other plots] ...

def plot_misclassified_samples(test_dataset, all_labels, all_preds, model_name, max_samples=5):
    misclassified_indices = [i for i, (label, pred) in enumerate(zip(all_labels, all_preds)) if label != pred]
    if not misclassified_indices:
        print("No misclassified samples!")
        return
    plot_samples_by_index(test_dataset, misclassified_indices[:max_samples], model_name, "misclassified")

# 3. FIX: Entirely rewritten to correctly fetch specific images from the dataset using global indices
def plot_samples_by_index(test_dataset, indices, model_name, tag):
    if len(indices) == 0:
        return
        
    plt.figure(figsize=(15, 6))
    
    for plot_idx, global_idx in enumerate(indices):
        # Fetch the specific image and label from the raw dataset
        image_tensor, true_label = test_dataset[global_idx]
        
        # Denormalize the image if necessary (Assuming standard ImageNet stats, adjust if different)
        # image = image_tensor.numpy().transpose((1, 2, 0))
        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        # image = std * image + mean
        # image = np.clip(image, 0, 1)
        
        # Simple transpose if no normalization was used:
        image = np.transpose(image_tensor.cpu().numpy(), (1, 2, 0))
        
        plt.subplot(1, len(indices), plot_idx + 1)
        plt.imshow(image)
        plt.title(f"Index: {global_idx}\nTrue: {true_label}")
        plt.axis("off")

    plt.suptitle(f"{tag.capitalize()} Samples - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, f"{tag}_samples_{model_name}.png"))
    plt.close()

if __name__ == "__main__":
    results = {}

    # Make sure these filenames match what your train.py actually produced! 
    # (e.g., if you saved them as best_model_Adam_fold0.pth, update this list)
    models_to_test = ["Adam", "SGD"] 

    for opt in models_to_test:
        acc = evaluate_model(model_name=opt, visualize=True)
        results[opt] = acc * 100

    print("\n📊 Evaluation Results:")
    for opt, acc in results.items():
        print(f"{opt}: {acc:.2f}% Test Accuracy")

    best = max(results, key=results.get)
    print(f"\n🏆 Best Optimizer: {best} with Accuracy: {results[best]:.2f}%")

    # [Keep your final summary bar plots exactly as they are]