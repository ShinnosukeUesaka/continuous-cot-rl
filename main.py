import argparse

from load_data import load_countdown_dataset
from rewards import load_countdown_rewards

def main():
    parser = argparse.ArgumentParser(description="A simple command-line to train the model")

    # Adding arguments
    parser.add_argument("--task", type=str, required=True, help="Name of the task to execute")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Hugging Face model name")
    parser.add_argument("--logging_steps", type=int, default=10, help="Number of steps to log the training progress")
    parser.add_argument("--save_steps", type=int, default=50, help="Number of steps to save the model")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for the model")
    parser.add_argument("--output_dir", type=str, default="./models/ccot-v1", help="Output directory to save the model")

    # Parsing arguments
    args = parser.parse_args()

    print(f"Started RL finetuning {args.model} for task {args.task}")

    config = {
        "logging_steps": args.logging_steps,
        "save_steps": args.logging_steps,
        "learning_rate": args.logging_steps,
        "batch_size": args.logging_steps,
        "report_to": "wandb"
    }

    if args.task == "countdown":
        dataset = load_countdown_dataset()
        reward_functions = load_countdown_rewards()

        train_dataset = dataset["train"]
        test_dataset = dataset["validation"]


    trainer = GRPOTrainer(
            model=args.model,
            output_dir=args.output_dir,
            reward_funcs=reward_functions,
            args=config,
            train_dataset=train_dataset,
            eval_dataset=test_dataset
    )
    
    trainer.train()


if __name__ == "__main__":
    main()