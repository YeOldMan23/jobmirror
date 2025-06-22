# Get Spark Session
import pyspark
from pyspark.sql.dataframe import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import FloatType, StructType, StructField, StringType, ArrayType, BooleanType, IntegerType



import os
from datetime import datetime

from .gold_feature_extraction.extract_skills import create_hard_skills_general_column, create_hard_skills_specific_column, create_soft_skills_column
from .gold_feature_extraction.match_experience import process_gold_experience
from .gold_feature_extraction.process_label_data import process_labels
from .gold_feature_extraction.feature_match import match_employment_type, match_work_authorization, match_location_preference

from .gold_feature_extraction.extract_edu import extract_education_features

def read_silver_table(table_name : str, snapshot_date : datetime, spark : SparkSession):
    selected_date = str(snapshot_date.year) + "-" + str(snapshot_date.month)
    table_dir     = os.path.join("datamart", "silver", table_name, f"{selected_date}.parquet")
    return spark.read.parquet(table_dir)

def data_processing_gold_features(snapshot_date: datetime, spark : SparkSession) -> None:
    # Read silver table
    df = read_silver_table("combined_resume_jd", snapshot_date, spark)

    # Add skills
    print("Processing skills")
    df = create_hard_skills_general_column(df)
    df = create_hard_skills_specific_column(df)
    df = create_soft_skills_column(df)
    df = extract_education_features(df)


    # Add in job experience
    print("Processing Experience")
    df = process_gold_experience(df)
    
    # match preferences 
    df = match_employment_type(df)
    df = match_work_authorization(df)
    df = match_location_preference(df)
    
    # Process the label store
    df_labels = process_labels(df)
    
    # Select only relevant columns
    df = df.select(
        "resume_id", "job_id", "snapshot_date", # General
        "soft_skills_mean_score", "soft_skills_max_score", "soft_skills_count", "soft_skills_ratio", # Soft skills
        "hard_skills_general_count", "hard_skills_general_ratio", "hard_skills_specific_count", "hard_skills_specific_ratio" # Hard skills
        "edu_gpa", # The standardized GPA
        "institution_tier"
        "work_authorization_match","employment_type_match","location_preference_match", # Match preferences
        "relevant_yoe", "total_yoe", "avg_exp_sim", "max_exp_sim", "is_freshie", # Experience data
        )
    
    df_labels = df_labels.select("resume_id", "job_id", "snapshot_date", "fit_label")

    # Save the parquet 
    print("Saving Feature Store")
    selected_date = str(snapshot_date.year) + "-" + str(snapshot_date.month)
    filename    = selected_date + ".parquet"
    output_path = os.path.join("datamart", "gold", "feature_store", filename)
    df.write.mode("overwrite").parquet(output_path)

    # Save the label store
    print("Saving Label Store")
    selected_date = str(snapshot_date.year) + "-" + str(snapshot_date.month)
    filename    = selected_date + ".parquet"
    output_path = os.path.join("datamart", "gold", "label_store", filename)
    df_labels.write.mode("overwrite").parquet(output_path)