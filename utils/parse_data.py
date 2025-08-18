# Basic imports
import os
import numpy as np
import sys
import json
from dotenv import load_dotenv

from datasets import load_dataset

load_dotenv()

"""
Parse the data from the online into local repository
"""

def get_hugging_face_dataset(save_loc : str):
    resume_dataset = load_dataset("cnamuangtoun/resume-job-description-fit")

    # Get the train and test data
    train_data = dataset["train"].to_pandas()
    test_data  = dataset["test"].to_pandas()


