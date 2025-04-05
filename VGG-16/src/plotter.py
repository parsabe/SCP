import evaluate
import os
import numpy as np
from config import PARAMS_PATH

def run_evaluation_only():
    results = {}

    for opt in ["Adam", "SGD"]:
        print(f"\n🔍 Checking for best model and parameters for: {opt}")

        param_path = os.path.join("/home/pb70gygu/Adam/best_params/", f"best_params_{opt}.npy")
        if not os.path.exists(param_path):
            print(f"⚠️ Best parameters file not found for {opt}: {param_path}")
            continue

        model_path = f"best_model_{opt}.pth"
        if not os.path.exists(model_path):
            print(f"⚠️ Trained model file not found for {opt}: {model_path}")
            continue

        print(f"✅ Found model and parameters for {opt}, starting evaluation...")
        acc = evaluate.evaluate_model(model_name=opt, visualize=True)
        results[opt] = acc * 100
        print(f"\n{opt} - Test Accuracy: {acc * 100:.2f}%")

    if results:
        print("\n📊 Final Evaluation Results:")
        for opt, acc in results.items():
            print(f"{opt}: {acc:.2f}% Test Accuracy")
    else:
        print("\n❌ No evaluations were performed. Please check model and parameter files.")

if __name__ == "__main__":
    run_evaluation_only()
