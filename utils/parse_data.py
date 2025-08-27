# Basic imports
import os
import numpy as np
import shutil
from tqdm import tqdm

import pandas as pd
from pandas import Timestamp

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

def split_save_files(df : pd.DataFrame,
                     save_dir : str,
                     no_cache : bool) -> None:
    """
    Split the dataset into resume, JD and labels
    """
    # Directories to conside
    jd_dir = os.path.join(save_dir, "job_description")
    resume_dir = os.path.join(save_dir, "resume")
    label_dir = os.path.join(save_dir, "label")
    
    # Clean the cache 
    if no_cache:
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

    # Make the new dirs
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        os.mkdir(jd_dir)
        os.mkdir(resume_dir)
        os.mkdir(label_dir)

    # Iterate through the rows and save the data
    for index, row in tqdm(df.iterrows(), total=len(df)):
        resume_file_name = f"{index}_resume.txt"
        jd_file_name = f"{index}_jd.txt"
        label_file_name = f"{index}_label.txt"

        with open(os.path.join(resume_dir, resume_file_name), "w") as resume_f:
            resume_f.write(row['resume_text'])

        with open(os.path.join(resume_dir, jd_file_name), "w") as jd_f:
            jd_f.write(row['job_description_text'])

        with open(os.path.join(resume_dir, label_file_name), "w") as label_f:
            label_f.write(row['label'])
