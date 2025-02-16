import re

from trl import apply_chat_template, GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer
from datasets import load_dataset
import datasets


class Trainer:
    def __init__(self, config: GRPOConfig, dataset_path: str, model_name: str):
        self.config = config
        self.model_name = model_name
        
        # Load and preprocess dataset
        self.dataset = datasets.load_from_disk(dataset_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.dataset = self.dataset.map(apply_chat_template, fn_kwargs={"tokenizer": self.tokenizer})
        
        self.train_dataset = self.dataset['train']
        self.test_dataset = self.dataset['test']
        
        # Initialize Trainer
        self.trainer = GRPOTrainer(
            model=self.model_name,
            args=self.config,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset
        )
    
    def train(self):
        self.trainer.train()