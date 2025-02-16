import argparse

def main():
    parser = argparse.ArgumentParser(description="A simple command-line to train the model")

    # Adding arguments
    parser.add_argument("--task", type=str, required=True, help="Name of the task to execute")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Hugging Face model name")

    # Parsing arguments
    args = parser.parse_args()

    print(f"Started RL finetuning {args.model} for task {args.task}")

    if args.task == "countdown":
        from src.countdown_task.train_countdown import Trainer
        

if __name__ == "__main__":
    main()
