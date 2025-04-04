import tune
import train
import evaluate
import config
import numpy as np
import os

def adam_files_exist():
    files = [
        "training_history_Adam_fold_0.npy",
        "training_history_Adam_fold_1.npy",
        "training_history_Adam_fold_2.npy",
        "best_model_Adam.pth",
        os.path.join(config.PARAMS_PATH, "best_params_Adam.npy")
    ]
    return all(os.path.exists(f) for f in files)

def run_training_with_best(opt_type, best_config):
    print(f"\nStarting Training with Best Hyperparameters for {opt_type}...")
    trainer = train.Trainer(
        learning_rate=best_config["learning_rate"],
        batch_size=best_config["batch_size"],
        optimizer_type=opt_type
    )
    best_val_acc = trainer.train(return_best_val_acc=True)
    print(f"\n{opt_type} - Best Validation Accuracy: {best_val_acc:.2f}%")

def main():
    results = {}

    if adam_files_exist():
        print("✅ All required Adam artifacts found. Skipping Adam tuning and training.")
        start_from = "SGD"
    else:
        print("🔁 Adam artifacts not complete. Starting full tuning & training for Adam.")
        start_from = "Adam"

    for opt_type in ["Adam", "SGD"]:
        if opt_type == "Adam" and start_from != "Adam":
            print(f"\nSkipping {opt_type} as it's already done.")
            continue

        print(f"\n🔍 Checking Best Hyperparameters for {opt_type}...")
        param_path = os.path.join(config.PARAMS_PATH, f"best_params_{opt_type}.npy")

        if os.path.exists(param_path):
            print(f"✅ Best parameters found for {opt_type}.")
            best_params = np.load(param_path, allow_pickle=True).item()
        else:
            print(f"⚙️ Best parameters not found for {opt_type}. Running hyperparameter tuning...")
            tune.run_hyperparameter_tuning(opt_type)
            best_params = np.load(param_path, allow_pickle=True).item()

        best_config = best_params["best_config"]
        print(f"📌 Best Hyperparameters for {opt_type}: {best_config}")

        model_path = f"best_model_{opt_type}.pth"
        if os.path.exists(model_path):
            print(f"✅ Model already trained for {opt_type}. Skipping training...")
        else:
            run_training_with_best(opt_type, best_config)

    # Final Evaluation
    print("\n📊 Generating Training and Test Evaluation Plots...")
    history = evaluate.load_training_history()
    evaluate.plot_training_curves(history)
    evaluate.evaluate_and_plot_test()
    print("\n✅ All plots saved in:", config.PLOT_PATH)


if __name__ == "__main__":
    main()
