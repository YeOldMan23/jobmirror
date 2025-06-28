# Get Spark Session
import pyspark
from pyspark.sql.dataframe import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import FloatType, StructType, StructField, StringType, ArrayType, BooleanType, IntegerType

import os
import argparse
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from mongodb_utils import get_pyspark_session
from date_utils import get_snapshot_dates

# from utils.s3_utils import upload_to_s3, read_parquet_from_s3
from gold_feature_extraction.extract_skills import create_hard_skills_general_column, create_hard_skills_specific_column, create_soft_skills_column
from gold_feature_extraction.feature_match import match_employment_type, match_work_authorization, match_location_preference
from gold_feature_extraction.process_label_data import process_labels
from gold_feature_extraction.extract_edu import extract_education_features
from gold_feature_extraction.match_experience import process_gold_experience


def read_silver_table(table_name: str, snapshot_date, spark, type: str):
    # Define project root
    project_root = Path("/opt/airflow")  # Adjust if needed

    # Format the snapshot date
    selected_date = f"{snapshot_date.year}-{snapshot_date.month:02}"

    # Construct the full file path
    table_path = project_root / "datamart/silver" / table_name / f"{selected_date}.parquet"

    # Read and return the DataFrame
    return spark.read.parquet(str(table_path))

# For S3
# def read_silver_table(table_name : str, snapshot_date : datetime, spark : SparkSession):
#     selected_date = str(snapshot_date.year) + "-" + str(snapshot_date.month)
#     table_dir = f"datamart/silver/{table_name}/{selected_date}.parquet"
#     return read_parquet_from_s3(spark, table_dir)

def data_processing_gold_features(snapshot_date: datetime, type, spark : SparkSession) -> None:
    
    # Connect to Google Drive
    # service = connect_to_gdrive()
    # parent_root = '1_eMgnRaFtt-ZSZD3zfwai3qlpYJ-M5C6' 
    # directory_path = ['datamart', 'gold', 'feature_store']
    # directory_id = get_folder_id_by_path(service, directory_path, parent_root)
    
    # Read silver table
    df = read_silver_table("combined_resume_jd", snapshot_date, spark, type)

    # Add skills
    print("Processing Skills")
    df = create_hard_skills_general_column(df)
    df = create_hard_skills_specific_column(df)
    df = create_soft_skills_column(df)

    # Add education
    print("Processing Education")
    df = extract_education_features(df)

    # Add in job experience
    print("Processing Experience")
    df = process_gold_experience(df)
    
    # match preferences 
    print("Processing Preferences")
    df = match_employment_type(df)
    df = match_work_authorization(df)
    df = match_location_preference(df)

    # Select only relevant columns
    df = df.select(
        "resume_id", "job_id", "snapshot_date", # General
        "soft_skills_mean_score", "soft_skills_max_score", "soft_skills_count", "soft_skills_ratio", # Soft skills
        "hard_skills_general_count", "hard_skills_general_ratio", "hard_skills_specific_count", "hard_skills_specific_ratio", # Hard skills
        "edu_level_match", "edu_level_score", "edu_field_match", "cert_match", "edu_gpa", "institution_tier", # Education features
        "work_authorization_match","employment_type_match","location_preference_match", # Match preferences
        "relevant_yoe", "total_yoe", "avg_exp_sim", "max_exp_sim", "is_freshie" # Experience data
        )
    # df_labels = df_labels.select("resume_id", "job_id", "snapshot_date", "fit_label")
    # Save the parquet 
    print("Saving Feature Store")

    selected_date = str(snapshot_date.year) + "-" + str(snapshot_date.month)
    filename    = selected_date + ".parquet"
    project_root = Path("/opt/airflow")
    # if type == "training":
    output_path = project_root / "datamart/gold/feature_store" / filename
    # elif type == "inference":
    #     output_path = project_root / "datamart/gold/online/feature_store" / filename

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save DataFrame
    df.write.mode("overwrite").parquet(str(output_path))

    # uploading to s3
    # s3_key = f"datamart/gold/feature_store/{filename}"
    # upload_to_s3(output_path, s3_key)
    print(f"Saved Gold Features : {selected_date} No. Rows : {df.count()}")

def data_processing_gold_labels(snapshot_date: datetime, spark : SparkSession, type) -> None:

    # Connect to Google Drive
    # service = connect_to_gdrive()
    # parent_root = '1_eMgnRaFtt-ZSZD3zfwai3qlpYJ-M5C6' 
    # directory_path = ['datamart', 'gold', 'label_store']
    # directory_id = get_folder_id_by_path(service, directory_path, parent_root)

    # Read silver table
    df = read_silver_table("combined_resume_jd", snapshot_date, spark, type)

    # Process the label store
    df_labels = process_labels(df)
    df_labels = df_labels.select("resume_id", "job_id", "snapshot_date", "fit_label")

    # Save the label store
    print("Saving Label Store")

    project_root = Path("/opt/airflow")

    # Format the date and filename
    selected_date = f"{snapshot_date.year}-{snapshot_date.month:02}"
    filename = f"{selected_date}.parquet"

    # Build the full output path
    output_path = project_root / "datamart/gold/label_store" / filename
 
    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_labels.write.mode("overwrite").parquet(str(output_path))

    # Upload parquet to drive
    # upload_file_to_drive(service, output_path, directory_id)

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument('--type', type=str, help='Inference or training')
    parser.add_argument('--store', type=str, required = True, help='feature or label')
    args = parser.parse_args()

    snapshot_date = datetime.strptime(args.snapshotdate, "%Y-%m-%d")

    try:

        load_dotenv("/opt/airflow/.env")
        # os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

        # Get the pyspark session
        spark = get_pyspark_session()

        # process gold feature and label stores
        if args.store == "feature":
            data_processing_gold_features(snapshot_date, args.type, spark)
        elif args.store == "label":
            data_processing_gold_labels(snapshot_date, args.type, spark)

    except Exception as e:
        print("An error occurred:", e)
    

