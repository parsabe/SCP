import numpy as np
import os
from train import Trainer
from config import PARAMS_PATH

def run_hyperparameter_tuning():
   
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

    for opt_type, configs in param_space.items():
        best_acc = 0
        best_config = None

        for config in configs:
            print(f"\nTrying Hyperparameters: {config} with Optimizer: {opt_type}")
            trainer = Trainer(
                learning_rate=config["learning_rate"],
                batch_size=config["batch_size"],
                optimizer_type=opt_type
            )
            val_acc = trainer.train(return_best_val_acc=True)

            if val_acc > best_acc:
                best_acc = val_acc
                best_config = config

     
        result = {"best_acc": best_acc, "best_config": best_config}
        np.save(os.path.join(PARAMS_PATH, f"best_params_{opt_type}.npy"), result)

        print(f"\n Best Hyperparameters for {opt_type}: {best_config} with Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    run_hyperparameter_tuning()
