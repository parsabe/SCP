import numpy as np
import os
from train import Trainer
from config import PARAMS_PATH

def run_hyperparameter_tuning(optimizer_type="Adam"):
    print(f"\n📣 Starting hyperparameter tuning for optimizer: {optimizer_type}")

    os.makedirs(PARAMS_PATH, exist_ok=True)

    param_space = {
        "Adam": [
            {"learning_rate": 0.001, "batch_size": 64},
            {"learning_rate": 0.0005, "batch_size": 64},
            {"learning_rate": 0.0005, "batch_size": 128}
        ],
        "SGD": [
            {"learning_rate": 0.01, "batch_size": 32},
            {"learning_rate": 0.005, "batch_size": 64},
            {"learning_rate": 0.001, "batch_size": 128}
        ]
    }

    configs = param_space.get(optimizer_type)
    if configs is None:
        raise ValueError(f"❌ Unknown optimizer type: {optimizer_type}")

    best_acc = 0
    best_config = None

    for config in configs:
        print(f"\n🔍 Trying Hyperparameters: {config} with Optimizer: {optimizer_type}")
        trainer = Trainer(
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
            optimizer_type=optimizer_type
        )
        val_acc = trainer.train(return_best_val_acc=True)

        if val_acc > best_acc:
            best_acc = val_acc
            best_config = config

    result = {
        "best_acc": best_acc,
        "best_config": best_config
    }

    param_save_path = os.path.join(PARAMS_PATH, f"best_params_{optimizer_type}.npy")
    np.save(param_save_path, result)

    print(f"\n✅ Best Hyperparameters for {optimizer_type}: {best_config} with Accuracy: {best_acc:.2f}%")
    print(f"📁 Saved to: {param_save_path}")


if __name__ == "__main__":
    run_hyperparameter_tuning("Adam")  # For standalone testing
