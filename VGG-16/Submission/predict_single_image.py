import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

from model import SimplifiedVGG
from config import DEVICE, NUM_CLASSES
from augmentation import produce_image

IMAGE_PATH = input("Enter the path of the image file: ")
IMAGE_SIZE = 56
MODEL_PATH = "best_model_Adam.pth"
CLASS_NAMES = [f"Class {i}" for i in range(NUM_CLASSES)]

tensor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def predict_image(image_path):
    raw_img = Image.open(image_path).convert("RGB")
    rng = np.random.default_rng(seed=1)
    processed_img, _ = produce_image(raw_img, resolution=IMAGE_SIZE, rng=rng)
    input_tensor = tensor_transform(processed_img).unsqueeze(0).to(DEVICE)

    model = SimplifiedVGG(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs).item()

    plt.imshow(raw_img)
    plt.title(f"Predicted: {CLASS_NAMES[predicted_class]} ({confidence*100:.1f}%) - Adam")
    plt.axis("off")
    plt.tight_layout()

    image_basename = os.path.basename(image_path).split('.')[0]
    output_path = f"{image_basename}_prediction.png"
    plt.savefig(output_path)
    print(f"Prediction saved as {output_path}")

predict_image(IMAGE_PATH)
