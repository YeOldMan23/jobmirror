# Get Spark Session
import pyspark
from pyspark.sql.dataframe import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import FloatType, StructType, StructField, StringType, ArrayType, BooleanType, IntegerType

import os
import argparse
from datetime import datetime
from utils.mongodb_utils import get_pyspark_session
from utils.date_utils import get_snapshot_dates

from .s3_utils import upload_to_s3, read_parquet_from_s3
from .gold_feature_extraction.extract_skills import create_hard_skills_general_column, create_hard_skills_specific_column, create_soft_skills_column
from .gold_feature_extraction.feature_match import match_employment_type, match_work_authorization, match_location_preference
from .gold_feature_extraction.process_label_data import process_labels

def read_silver_table(table_name : str, snapshot_date : datetime, spark : SparkSession, type):
    selected_date = str(snapshot_date.year) + "-" + str(snapshot_date.month)
    if type == "training":
        table_dir  = os.path.join("datamart", "silver", table_name, f"{selected_date}.parquet")
    elif type == "inference":
        table_dir     = os.path.join("datamart", "silver", "online", table_name, f"{selected_date}.parquet")
    return spark.read.parquet(table_dir)

# For S3
# def read_silver_table(table_name : str, snapshot_date : datetime, spark : SparkSession):
#     selected_date = str(snapshot_date.year) + "-" + str(snapshot_date.month)
#     table_dir = f"datamart/silver/{table_name}/{selected_date}.parquet"
#     return read_parquet_from_s3(spark, table_dir)

def data_processing_gold_features(snapshot_date: datetime, type, spark : SparkSession) -> None:
    # Read silver table
    df = read_silver_table("combined_resume_jd", snapshot_date, spark, type)

    # Add skills
    df = create_hard_skills_general_column(df)
    df = create_hard_skills_specific_column(df)
    df = create_soft_skills_column(df)

    # match preferences 
    df = match_employment_type(df)
    df = match_work_authorization(df)
    df = match_location_preference(df)

    # Process the label store
    df = process_labels(df)

    # Select only relevant columns
    df = df.select(
        "resume_id", "job_id", "snapshot_date", # General
        "soft_skills_mean_score", "soft_skills_max_score", "soft_skills_count", "soft_skills_ratio", # Soft skills
        "hard_skills_general_count", "hard_skills_general_ratio", "hard_skills_specific_count", "hard_skills_specific_ratio", # General skills
        "work_authorization_match","employment_type_match","location_preference_match",
        # Add your new columns here
        "relevant_yoe", "total_yoe", "avg_exp_sim", "max_exp_sim", "is_freshie", # Experience data
        "fit_label" # Label data
        )
    df_labels = df_labels.select("resume_id", "job_id", "snapshot_date", "fit_label")

    # Save the parquet 
    selected_date = str(snapshot_date.year) + "-" + str(snapshot_date.month)
    filename    = selected_date + ".parquet"
    if type == "training":
        output_path = os.path.join("datamart", "gold", "feature_store", filename)
    elif type == "inference":
        output_path = os.path.join("datamart", "gold", "online", "feature_store", filename)
    df.write.mode("overwrite").parquet(output_path)

    # Save the label store
    print("Saving Label Store")
    selected_date = str(snapshot_date.year) + "-" + str(snapshot_date.month)
    filename    = selected_date + ".parquet"
    if type == "training":
        output_path = os.path.join("datamart", "gold", "label_store", filename)
    elif type == "inference":
        output_path = os.path.join("datamart", "gold", "online", "label_store", filename)
    df_labels.write.mode("overwrite").parquet(output_path)

    # uploading to s3
    # s3_key = f"datamart/gold/feature_store/{filename}"
    # upload_to_s3(output_path, s3_key)
    print(f"Saved Gold Features : {selected_date} No. Rows : {df.count()}")

if __name__ == "__main__":
    # Get the pyspark session
    spark = get_pyspark_session()
    
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args('--type', type=int, default=1, help='Inference or training')
    args = parser.parse_args()

    # process gold table
    data_processing_gold_features(args.snapshotdate, args.type, spark)
    
    # # Datamart dir
    # datamart_dir = os.path.join(os.getcwd(), "datamart")

    # # Get the range of dates
    # date_range = get_snapshot_dates(datetime(2021, 6, 1), datetime(2021, 7, 31))

    # # For each range, read the silver table and parse
    # for cur_date in date_range:
    #     snapshot_date = f"{cur_date.year}-{cur_date.month}"
    #     print("Processing gold {}".format(snapshot_date))
    #     data_processing_gold_features(cur_date, spark)