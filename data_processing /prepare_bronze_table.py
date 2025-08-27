"""
Prepare the bronze table
"""
import os
import sys
import pandas as pd
import shutil

import pyspark
from pyspark.sql.session import SparkSession

# Change the sys to one level down
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.parse_data import *

# Do regular imports
from llama_cpp import Llama

def prepare_bronze_table(spark : SparkSession, 
                         save_dir : str,
                         no_cache : bool):
    """
    Prepare the bronze table to be saved
    """
    bronze_loc = os.path.join(save_dir, "bronze")

    # Remove the old location if available 
    if no_cache:
        if os.path.exists(bronze_loc):
            shutil.rmtree(bronze_loc)

    # Make the new dir if available
    if not os.path.exists(bronze_loc):
        os.mkdir(bronze_loc)

        # Make the sub_dirs
        os.mkdir(os.path.join(bronze_loc, "resume"))
        os.mkdir(os.path.join(bronze_loc, "job_description"))
        os.mkdir(os.path.join(bronze_loc, "label"))

    print("---Downloading Dataset---")
    



