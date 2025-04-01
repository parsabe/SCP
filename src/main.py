import tune
import train
import evaluate
import config
import numpy as np
import os

def main():
    results = {}

    for opt_type in ["Adam", "SGD"]:
        print(f"\n Checking Best Hyperparameters for {opt_type}...")

        
        param_path = os.path.join(config.PARAMS_PATH, f"best_params_{opt_type}.npy")
        if os.path.exists(param_path):
            print(f" Best parameters found for {opt_type}.")
            best_params = np.load(param_path, allow_pickle=True).item()
        else:
            print(f" Best parameters not found for {opt_type}. Running hyperparameter tuning...")
            tune.run_hyperparameter_tuning()
            best_params = np.load(param_path, allow_pickle=True).item()

        best_config = best_params["best_config"]
        print(f" Best Hyperparameters for {opt_type}: {best_config}")

        model_path = f"best_model_{opt_type}.pth"
        if os.path.exists(model_path):
            print(f" Model already trained for {opt_type}. Skipping training...")
        else:
            print(f"\n Starting Training with Best Hyperparameters for {opt_type}...")
            trainer = train.Trainer(
                learning_rate=best_config["learning_rate"],
                batch_size=best_config["batch_size"],
                optimizer_type=opt_type
            )
            best_val_acc = trainer.train(return_best_val_acc=True)
            print(f"\n{opt_type} - Best Validation Accuracy: {best_val_acc:.2f}%")

        
        print(f"\n Starting Evaluation for {opt_type}...")
        test_acc = evaluate.evaluate_model(model_name=opt_type, visualize=True)
        results[opt_type] = test_acc * 100
        print(f"\n{opt_type} - Test Accuracy: {test_acc * 100:.2f}%")


    if results:
        print("\nFinal Evaluation Summary:")
        for opt, acc in results.items():
            print(f"{opt}: {acc:.2f}% Test Accuracy")

        best = max(results, key=results.get)
        print(f"\n Best Optimizer based on Test Accuracy: {best} with Accuracy: {results[best]:.2f}%")
    else:
        print("\n No evaluations were performed. Please ensure best_params and model files exist.")

if __name__ == "__main__":
    main()
