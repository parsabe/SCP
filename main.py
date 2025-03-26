import train
import evaluate

if __name__ == "__main__":
    print("Starting Training...")
    trainer = train.Trainer()
    trainer.train()

    user_input = input("Do you want to evaluate the model? (yes/no): ").strip().lower()
    if user_input == "yes":
        visualize_input = input("Do you want visualizations? (yes/no): ").strip().lower()
        evaluate.evaluate_model(visualize=(visualize_input == "yes"))
