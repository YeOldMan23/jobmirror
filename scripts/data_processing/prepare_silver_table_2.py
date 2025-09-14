"""
Silver table 2 processes the table made from silver table 1, processing the data and turning
it into actual columns
"""

import pyspark
from pyspark.sql.session import SparkSession
import pandas as pd

import os
import sys
import shutil
import json

def prepare_silver_table_2(save_dir : str,
                         snapshot_date : str,
                         no_cache : bool,
                         spark : SparkSession):
    
    pass