"""
Prepare the silver table from the bronze table
1) Combine all the data from all jsons into a single table
2) Convert the table to Pyspark Dataframe
"""
import pyspark
from pyspark.sql.session import SparkSession
import pandas as pd

import os
import sys
import shutil
import json

def prepare_silver_table_1(save_dir : str,
                         snapshot_date : str,
                         no_cache : bool,
                         spark : SparkSession):
    year = snapshot_date.split("_")[0]
    month = snapshot_date.split("_")[1]

    bronze_loc = os.path.join(save_dir, "bronze", f"{year}_{month}")
    silver_loc = os.path.join(save_dir, "silver", f"{year}_{month}")
    
    if no_cache:
        if os.path.exists(silver_loc):
            shutil.rmtree(silver_loc)

    if not os.path.exists(silver_loc):
        os.mkdir(silver_loc)

        os.mkdir(os.path.join(silver_loc, "resume"))
        os.mkdir(os.path.join(silver_loc, "job_description"))
        os.mkdir(os.path.join(silver_loc, "label"))

    # Read the data
    resume_dir = os.path.join(bronze_loc, "resume")
    jd_dir = os.path.join(bronze_loc, "job_description")
    label_dir = os.path.join(bronze_loc, "label")

    resume_list = os.listdir(resume_dir)

    # Parse through all the files, make it into a PD
    for resume_item in resume_list:
        resume_number = resume_item.split("_")[0]
        jd_item = resume_number + "_job_description.json"
        label_item = resume_number + "_label.json"

        # Read all the files
        with open(os.path.join(resume_dir, resume_item), "r") as resume_f:
            resume_data = json.load(resume_f)

        with open(os.path.join(jd_dir, jd_item), "r") as jd_f:
            jd_data = json.load(resume_f)

        with open(os.path.join(label_dir, label_item), "r") as label_f:
            label_data = json.load(resume_f)

        # 
        

    pass