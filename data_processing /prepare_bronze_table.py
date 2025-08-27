"""
Prepare the Bronze Table After basic processing
"""
import os
import sys
import json
import shutil

import pyspark
from pyspark.sql.session import SparkSession

# Do regular imports
from llama_cpp import Llama

# Change the sys to one level down
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.parse_data import *
from utils.llm_methods import *

def prepare_bronze_table(spark : SparkSession,
                         datamart_dir : str,
                         bronze_dir : str)