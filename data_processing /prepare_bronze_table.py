"""
Prepare the Bronze Table After basic processing
"""
import os
import sys
import json
import shutil
from tqdm import tqdm

import pyspark
from pyspark.sql.session import SparkSession

# Do regular imports
from llama_cpp import Llama

# Change the sys to one level down
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.parse_data import *
from utils.llm_methods import *

def prepare_bronze_table(spark : SparkSession,
                         save_dir : str,
                         no_cache : bool) -> None:
    """
    Prepare the bronze table by parsing the data through a LLM
    """
    datamart_dir = os.path.join(save_dir, "datamart")
    train_datamart_dir = os.path.join(datamart_dir, "train")
    test_datamart_dir = os.path.join(datamart_dir, "test")

    bronze_dir = os.path.join(save_dir, "bronze")
    train_bronze_dir = os.path.join(bronze_dir, "train")
    test_bronze_dir = os.path.join(bronze_dir, "test")

    if no_cache:
        if os.path.exists(bronze_dir):
            shutil.rmtree(bronze_dir)

    # Make the new directory 
    if not os.path.exists(bronze_dir):
        os.mkdir(bronze_dir)

        # Make the dates
        for i in range(1, 11):
            cur_date = f"2021_{i}"

            os.mkdir(os.path.join(train_bronze_dir, cur_date, "resume"))
            os.mkdir(os.path.join(train_bronze_dir, cur_date, "job_description"))
            os.mkdir(os.path.join(train_bronze_dir, cur_date, "label"))

        for i in range(11, 13):
            cur_date = f"2021_{i}"

            os.mkdir(os.path.join(test_bronze_dir, cur_date, "resume"))
            os.mkdir(os.path.join(test_bronze_dir, cur_date, "job_description"))
            os.mkdir(os.path.join(test_bronze_dir, cur_date, "label"))

    # Read the data from each directory, and parse the data (train first)
    for i in range(1, 11):
        cur_date = f"2021_{i}"

        print(f"Parsing Training Bronze Table Date {cur_date}")

        train_dataset_resume_loc = os.path.join(train_datamart_dir, cur_date, "resume")
        train_dataset_jd_loc = os.path.join(train_datamart_dir, cur_date, "job_description")
        train_dataset_label_loc = os.path.join(train_datamart_dir, cur_date, "label")

        train_data_resume_file_list = os.listdir(train_dataset_resume_loc)
        train_data_jd_file_list = os.listdir(train_dataset_jd_loc)
        train_data_label_file_list = os.listdir(train_dataset_label_loc)

        print("---Preparing Bronze Train Resume Data---")
        for resume_file in tqdm(train_data_resume_file_list):
            cur_save_file = os.path.join(train_bronze_dir, cur_date, "resume", resume_file)
            cur_read_file = os.path.join(train_dataset_resume_loc, resume_file)
            parse_w_llm_and_save_data(cur_read_file, cur_save_file)
        
        print("---Preparing Bronze Train JD Data---")
        for jd_file in tqdm(train_data_jd_file_list):
            cur_save_file = os.path.join(train_bronze_dir, cur_date, "job_description", jd_file)
            cur_read_file = os.path.join(train_dataset_jd_loc, jd_file)
            parse_w_llm_and_save_data(cur_read_file, cur_save_file)

        print("---Preparing Bronze Train Label Data---")
        for label_file in tqdm(train_data_label_file_list):
            cur_save_file = os.path.join(train_bronze_dir, cur_date, "job_description", label_file)
            cur_read_file = os.path.join(train_dataset_label_loc, label_file)
            parse_w_llm_and_save_data(cur_read_file, cur_save_file)

    # Now do the test data
    for i in range(11, 13):
        cur_date = f"2021_{i}"

        print(f"Parsing Testing Bronze Table Date {cur_date}")

        test_dataset_resume_loc = os.path.join(test_datamart_dir, cur_date, "resume")
        test_dataset_jd_loc = os.path.join(test_datamart_dir, cur_date, "job_description")
        test_dataset_label_loc = os.path.join(test_datamart_dir, cur_date, "label")

        test_data_resume_file_list = os.listdir(test_dataset_resume_loc)
        test_data_jd_file_list = os.listdir(test_dataset_jd_loc)
        test_data_label_file_list = os.listdir(test_dataset_label_loc)
        
        print("---Preparing Bronze Test Resume Data---")
        for resume_file in tqdm(test_data_resume_file_list):
            cur_save_file = os.path.join(test_bronze_dir, cur_date, "resume", resume_file)
            cur_read_file = os.path.join(test_dataset_resume_loc, resume_file)
            parse_w_llm_and_save_data(cur_read_file, cur_save_file)
        
        print("---Preparing Bronze Test JD Data---")
        for jd_file in tqdm(test_data_jd_file_list):
            cur_save_file = os.path.join(test_bronze_dir, cur_date, "job_description", jd_file)
            cur_read_file = os.path.join(test_dataset_jd_loc, jd_file)
            parse_w_llm_and_save_data(cur_read_file, cur_save_file)

        print("---Preparing Bronze Test Label Data---")
        for label_file in tqdm(test_data_label_file_list):
            cur_save_file = os.path.join(test_bronze_dir, cur_date, "job_description", label_file)
            cur_read_file = os.path.join(test_dataset_label_loc, label_file)
            parse_w_llm_and_save_data(cur_read_file, cur_save_file)

    print("---Job Complete---")

