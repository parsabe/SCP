from config import torch, sns, plt, confusion_matrix, DEVICE, NUM_CLASSES, load_dataset
from model import SimplifiedVGG

def evaluate_model(visualize=True):
    model = SimplifiedVGG().to(DEVICE)
    model.load_state_dict(torch.load("trained_model.pth"))
    model.eval()

    _, test_loader = load_dataset.load_dataset(batch_size=32, path="./data")

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE).long()
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if visualize:
        visualize_results(all_labels, all_preds)

def visualize_results(all_labels, all_preds):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

if __name__ == "__main__":
    user_input = input("Do you want visualizations? (yes/no): ").strip().lower()
    evaluate_model(visualize=(user_input == "yes"))
