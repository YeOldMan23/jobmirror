"""
Prepare the bronze table
"""
import os
import sys
import pandas as pd
import shutil
from datetime import datetime
import argparse

# Change the sys to one level down
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.parse_data import *
from utils.llm_methods import *

def prepare_datamart(save_dir : str,
                     no_cache : bool):
    """
    Prepare the bronze table to be saved
    """
    datamart_dir = os.path.join(save_dir, "datamart")
    train_datamart_dir = os.path.join(datamart_dir, "train")
    test_datamart_dir = os.path.join(datamart_dir, "test")

    # Remove the old location if available 
    if no_cache:
        if os.path.exists(datamart_dir):
            shutil.rmtree(datamart_dir)

    # Make the new dir if available
    if not os.path.exists(datamart_dir):
        os.mkdir(datamart_dir)

        # Make the sub_dirs
        for i in range(1, 11):
            cur_date = f"2021_{i}"

            train_date_dir = os.path.join(train_datamart_dir, cur_date)
            os.mkdir(train_date_dir)

            os.mkdir(os.path.join(train_date_dir, "resume"))
            os.mkdir(os.path.join(train_date_dir, "job_description"))
            os.mkdir(os.path.join(train_date_dir, "label"))

        for i in range(11, 13):
            cur_date = f"2021_{i}"

            test_date_dir = os.path.join(test_datamart_dir, cur_date)
            os.mkdir(test_date_dir)

            os.mkdir(os.path.join(test_date_dir, "resume"))
            os.mkdir(os.path.join(test_date_dir, "job_description"))
            os.mkdir(os.path.join(test_date_dir, "label"))


    print("---Downloading Dataset---")
    train_data, test_data = get_hugging_face_dataset()

    # Backdate the dates for the train data to be first 10 months of the year 2021, then test is the last 2 months
    train_start = Timestamp(datetime(2021, 1, 1))
    train_end   = Timestamp(datetime(2021, 10, 31))
    test_start  = Timestamp(datetime(2021, 11, 1))
    test_end    = Timestamp(datetime(2021, 12, 31))

    train_data = generate_random_data_dates(train_data, train_start, train_end)
    test_data = generate_random_data_dates(test_data, test_start, test_end)

    print("---Saving Datamart---")
    split_save_files(train_data, train_datamart_dir, no_cache)
    split_save_files(test_data, test_datamart_dir, no_cache)

    print("---Job Complete---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Datamart")
    parser.add_argument("--no_cache", type=bool, default=False, help="Make new datamart if True")
    args = parser.parse_args()

    save_dir = os.path.join(os.getcwd(), "..", "data")
    
    prepare_datamart(save_dir, args.no_cache)
    pass