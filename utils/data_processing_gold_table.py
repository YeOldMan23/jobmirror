# Get Spark Session
import pyspark
from pyspark.sql.dataframe import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import FloatType, StructType, StructField, StringType, ArrayType, BooleanType, IntegerType

import os
from datetime import datetime

from .gold_feature_extraction.extract_skills import create_hard_skills_general_column, create_hard_skills_specific_column, create_soft_skills_column

def read_silver_table(table_name : str, snapshot_date : datetime, spark : SparkSession):
    selected_date = str(snapshot_date.year) + "-" + str(snapshot_date.month)
    table_dir     = os.path.join("datamart", "silver", table_name, f"{selected_date}.parquet")
    return spark.read.parquet(table_dir)

def data_processing_gold_features(snapshot_date: datetime, spark : SparkSession) -> None:
    # Read silver table
    df = read_silver_table("combined", snapshot_date, spark)

    # Add skills
    df = create_hard_skills_general_column(df)
    df = create_hard_skills_specific_column(df)
    df = create_soft_skills_column(df)

    # Select only relevant columns
    df = df.select(
        "resume_id", "job_id", "snapshot_date", # General
        "soft_skills_mean_score", "soft_skills_max_score", "soft_skills_count", "soft_skills_ratio", # Soft skills
        "hard_skills_general_count", "hard_skills_general_ratio", "hard_skills_specific_count", "hard_skills_specific_ratio" # General skills
        # Add your new columns here
        )

    # Save the parquet 
    selected_date = str(snapshot_date.year) + "-" + str(snapshot_date.month)
    filename    = selected_date + ".parquet"
    output_path = os.path.join("datamart", "gold", "feature_store", filename)
    df.write.mode("overwrite").parquet(output_path)