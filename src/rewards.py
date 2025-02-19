import re

def load_countdown_rewards():

    # Define reward functions
    def format_reward_func(completions):
        """Reward function that checks if the completion has a specific format."""
        pattern = r"^<think>.*?</think><answer>.*?</answer>$"
        matches = [re.match(pattern, content) for content in completions]

        return [1.0 if match else 0.0 for match in matches]

    def reward_func(completions, ground_truth):
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
    
    return [format_reward_func, reward_func]