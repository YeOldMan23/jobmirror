# Basic imports
import os
import numpy as np

import pandas as pd
from pandas import Timestamp

import json
from dotenv import load_dotenv

from datasets import load_dataset

load_dotenv()

"""
Parse the data from the online into local repository
"""

def get_hugging_face_dataset():
    resume_jd_dataset = load_dataset("cnamuangtoun/resume-job-description-fit")

    # Get the train and test data
    train_data = resume_jd_dataset["train"].to_pandas()
    test_data  = resume_jd_dataset["test"].to_pandas()

    return train_data, test_data

def generate_random_data_dates(df : pd.DataFrame, 
                               start_date : Timestamp, 
                               end_date : Timestamp,
                               seed : int = 42) -> pd.DataFrame:
    """
    Generate Random dates for the dataframe, append to list
    """
    randomizer = np.random.default_rng(seed)

    random_train_dates = pd.to_datetime(
        randomizer.uniform(start_date.value, end_date.value, size=len(df))
    )
    df['snapshot_date'] = random_train_dates

    return df

