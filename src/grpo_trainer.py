import re

from utils import valid_models

import torch
from pydantic import BaseModel, Field, ValidationError, field_validator
from datasets import Dataset



class GRPOTrainer(BaseModel):
    # model name validation
    model: str
    output_dir: str
    reward_funcs: list = Field(..., min_items=1)
    args: dict
    train_dataset: Dataset = Field(..., description="Must be a Hugging Face Dataset object")
    test_dataset: Dataset = Field(..., description="Must be a Hugging Face Dataset object")


    @field_validator("train_dataset", "test_dataset")
    @classmethod
    def check_huggingface_dataset(cls, value):
        if not isinstance(value, Dataset):
            raise ValueError("The dataset field must be a Hugging Face Dataset object.")
        return value
    

    @field_validator("model")
    @classmethod
    def validate_model(cls, value):
        """Ensure the model name is valid based on `valid_models`."""
        valid_pattern = re.compile(f"^({'|'.join(map(re.escape, valid_models))})$")
        if not valid_pattern.match(value):
            raise ValueError(f"Invalid model name: {value}. Must be one of {valid_models}.")
        return value
    
    
    def train():
        raise NotImplementedError() 





