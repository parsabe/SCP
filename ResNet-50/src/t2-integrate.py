import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# Import your existing configurations
from config import DEVICE, NUM_CLASSES, load_dataset, PLOT_PATH, MODELS_PATH

def test_and_visualize(model_name="Adam", num_images=16):
    print(f"\n--- Running Visual Test for {model_name} ---")

    # 1. Load the ResNet-18 Architecture
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

    # 2. Find and Load the Best Saved Weights
    search_pattern = os.path.join(MODELS_PATH, f"best_model_{model_name}_fold*.pth")
    found_models = glob.glob(search_pattern)

    if not found_models:
        print(f"ERROR: Could not find model {search_pattern}")
        return

    model_path = found_models[0]
    print(f"Loaded weights from: {model_path}")

    # Strip the DataParallel wrapper if it exists
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model = model.to(DEVICE)
    model.eval()

    # 3. Grab a random batch of images from your EXISTING test dataset
    print("Loading test dataset...")
    _, test_loader = load_dataset.load_dataset(batch_size=32, path=r"C:\Users\parsa\Desktop\code\SCP\ResNet-50\augmented_data")
    
    # next(iter()) pulls exactly one batch of data (32 images)
    images, labels = next(iter(test_loader))
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    # 4. Make Predictions
    print("Making predictions...")
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    # 5. Build the Visual Grid
    images = images.cpu()
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    fig = plt.figure(figsize=(12, 12))
    
    # Calculate grid size (e.g., 4x4 for 16 images)
    rows = int(np.ceil(np.sqrt(num_images)))
    cols = int(np.ceil(num_images / rows))

    for i in range(min(num_images, len(images))):
        ax = fig.add_subplot(rows, cols, i + 1)
        
        # Convert PyTorch tensor (Color, Height, Width) to Matplotlib format (Height, Width, Color)
        img = images[i].numpy().transpose((1, 2, 0))
        
        # Clip values to [0, 1] so Matplotlib doesn't show weird colors due to normalization
        img = np.clip(img, 0, 1)

        ax.imshow(img)
        
        # Green text if correct, Red text if wrong
        color = "green" if preds[i] == labels[i] else "red"
        ax.set_title(f"True: {labels[i]}\nPred: {preds[i]}", color=color, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    
    # 6. Save the image to your plots folder
    os.makedirs(PLOT_PATH, exist_ok=True)
    save_file = os.path.join(PLOT_PATH, f"visual_grid_test_{model_name}.png")
    plt.savefig(save_file)
    plt.close()
    
    print(f"\nSUCCESS! Visual test complete.")
    print(f"Go open this file to see the results: {save_file}")

if __name__ == "__main__":
    # You can change this to "SGD" if you want to test the other model!
    test_and_visualize(model_name="Adam", num_images=16)