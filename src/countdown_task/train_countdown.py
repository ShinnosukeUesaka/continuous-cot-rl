import re
from trl import GRPOConfig
from src.trainer import Trainer

# Define reward functions
def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    matches = [re.match(pattern, content) for content in completions]

    return [1.0 if match else 0.0 for match in matches]

def reward_func(completions, ground_truth, **kwargs):
    """Reward function checking if the answer matches the ground truth."""
    matches = [re.search(r"<answer>(.*?)</answer>", completion) for completion in completions]
    contents = [match.group(1) if match else "" for match in matches]
    rewards = []

    for c, gt in zip(contents, ground_truth):
        try:
            calculation_result = eval(c)
            rewards.append(1.0 if calculation_result == gt else 0.0)
        except:
            rewards.append(0.0)
    return rewards


# Instantiate Trainer
training_args = GRPOConfig(output_dir="continuous-rl-v1", logging_steps=10, save_steps=50, report_to='wandb')
trainer_instance = Trainer(config=training_args, dataset_path="./countdown_tasks", model_name="Qwen/Qwen2.5-3B-Instruct")
trainer_instance.train()