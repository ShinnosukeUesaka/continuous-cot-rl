import os
from datasets import load_dataset
from litgpt import LLM

def load_countdown_dataset():
    # Load the raw dataset from the specified source
    raw_dataset = load_dataset('Jiayi-Pan/Countdown-Tasks-3to4', split='train')

    # Function to map each example to the desired prompt format
    def map_fn(example):
        prompt = [
            {"role": "system", "content": "You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer"},
            {"role": "user", "content": f"Using the numbers {example['nums']}, create an equation that equals {example['target']}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."}
        ]
        return {
            "prompt": prompt
        }

    # Apply the mapping function to the dataset
    raw_dataset = raw_dataset.map(map_fn)

    # Split the dataset into training and testing sets
    raw_dataset = raw_dataset.train_test_split(test_size=0.1)

    # Save the dataset to disk
    raw_dataset.save_to_disk("countdown_tasks")

    # Define the output directory path
    parent_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(parent_parent_dir, 'data', 'countdown')

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the dataset to the specified output directory
    raw_dataset.save_to_disk(output_dir)
    
    return raw_dataset


def load_pretrained_model(model_name):
    """
    Load the Qwen model from the specified path using litgpt.

    Args:
        model_path (str): The path to the Qwen model file.

    Returns:
        model: The loaded model.
    """

    # Get the parent directory of the current file
    path_models = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/models"
    return LLM.load(model=os.path.join(path_models, model_name))