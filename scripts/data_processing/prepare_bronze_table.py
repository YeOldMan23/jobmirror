"""
Prepare the Bronze Table After basic processing
"""
import os
import sys
import argparse
import shutil
from tqdm import tqdm

# import pyspark
# from pyspark.sql.session import SparkSession

# Do regular imports
from llama_cpp import Llama

# Change the sys to one level down
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.parse_data import *
from utils.llm_methods import *

def prepare_bronze_table(save_dir : str,
                         snapshot_date : str,
                         no_cache : bool) -> None:
    """
    Prepare the bronze table by parsing the data through a LLM
    """
    year = int(snapshot_date.split("_")[0])
    month = int(snapshot_date.split("_")[1])

    datamart_dir = os.path.join(save_dir, "datamart", f"{year}_{month}")
    bronze_dir      = os.path.join(save_dir, "bronze")
    bronze_date_dir = os.path.join(bronze_dir, f"{year}_{month}")

    if no_cache:
        if os.path.exists(bronze_dir):
            shutil.rmtree(bronze_dir)

    # Make the new directory 
    if not os.path.exists(bronze_dir):
        os.mkdir(bronze_dir)

    if not os.path.exists(bronze_date_dir):
        # Make the dates
        os.mkdir(bronze_date_dir)

        os.mkdir(os.path.join(bronze_date_dir, "resume"))
        os.mkdir(os.path.join(bronze_date_dir, "job_description"))
        os.mkdir(os.path.join(bronze_date_dir, "label"))

    # Read the data from each directory, and parse the data (train first)
    print(f"Parsing Training Bronze Table Date {year}_{month}")

    train_dataset_resume_loc = os.path.join(datamart_dir, "resume")
    train_dataset_jd_loc = os.path.join(datamart_dir, "job_description")
    train_dataset_label_loc = os.path.join(datamart_dir, "label")

    train_data_resume_file_list = os.listdir(train_dataset_resume_loc)
    train_data_jd_file_list = os.listdir(train_dataset_jd_loc)
    train_data_label_file_list = os.listdir(train_dataset_label_loc)

    print("---Preparing Bronze Train Resume Data---")
    for resume_file in tqdm(train_data_resume_file_list):
        json_format = resume_file.rstrip("txt") + "json"
        cur_save_file = os.path.join(bronze_date_dir, "resume", json_format)
        cur_read_file = os.path.join(train_dataset_resume_loc, resume_file)
        parse_w_llm_and_save_data(cur_read_file, cur_save_file)
    
    print("---Preparing Bronze Train JD Data---")
    for jd_file in tqdm(train_data_jd_file_list):
        json_format = jd_file.rstrip("txt") + "json"
        cur_save_file = os.path.join(bronze_date_dir, "job_description", json_format)
        cur_read_file = os.path.join(train_dataset_jd_loc, jd_file)
        parse_w_llm_and_save_data(cur_read_file, cur_save_file)

    print("---Preparing Bronze Train Label Data---")
    for label_file in tqdm(train_data_label_file_list):
        json_format = label_file.rstrip("txt") + "json"
        cur_save_file = os.path.join(bronze_date_dir, "label", json_format)
        cur_read_file = os.path.join(train_dataset_label_loc, label_file)
        parse_w_llm_and_save_data(cur_read_file, cur_save_file)

    print("---Job Complete---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Datamart")
    parser.add_argument("--snapshot_date", required=True, type=str, help="Date of snapshot")
    parser.add_argument("--no_cache", type=bool, default=False, help="Make new datamart if True")
    args = parser.parse_args()

    print("---Starting Job---")

    save_dir = os.path.join(os.getcwd(), "..", "data")
    prepare_bronze_table(save_dir, args.snapshot_date, args.no_cache)

    print("---Job Complete---")