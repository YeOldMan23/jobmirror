"""
Read the parquet files and extract new information out of them
"""


from mongodb_utils import read_bronze_resume_as_pyspark, read_bronze_jd_as_pyspark, read_bronze_labels_as_pyspark
# from gdrive_utils import connect_to_gdrive, get_folder_id_by_path, upload_file_to_drive
# from s3_utils import upload_to_s3, read_s3_data
from config import AWSConfig

# Get the features to match
from silver_feature_extraction.extract_exp import get_resume_yoe, get_title_similarity_score
from silver_feature_extraction.extract_edu import parse_education_udf_factory, determine_edu_mapping, determine_certification_types
from silver_feature_extraction.extract_skills import create_hard_skills_column, create_soft_skills_column
from silver_feature_extraction.extract_misc import clean_employment_type_column, location_lookup, standardize_location_column, clean_work_authorization_column, standardize_label

# Get Spark Session
import pyspark
from pyspark.sql.dataframe import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import FloatType, StructType, StructField, StringType, ArrayType, BooleanType, IntegerType
from mongodb_utils import get_pyspark_session
from dotenv import load_dotenv
from pathlib import Path

import os
# import shutil
import boto3
import argparse
import uuid
import tempfile

import torch
from datetime import datetime
from typing import Optional

# load_dotenv()
###################################################
# Gold Table Aggregations for Experience
###################################################

@udf(FloatType())
def get_relevant_yoe(sim_matrix, yoe_list, threshold : float):
    """
    Get the relevant YoE from the array
    """
    relevant_yoe = 0

    for cur_yoe, cur_exp_sim in zip(yoe_list, sim_matrix):
        if cur_exp_sim >= threshold:
            relevant_yoe += cur_yoe

    return max(0, relevant_yoe)

@udf(FloatType())
def get_total_yoe(yoe_list):
    """
    Get the total YoE from the array
    """
    return max(0, sum(yoe_list))

@udf(FloatType())
def get_avg_job_sim(sim_matrix):
    """
    Get Average Job Sim
    """
    if len(sim_matrix) > 0:
        return sum(sim_matrix) / len(sim_matrix)
    else:
        return 0
    
@udf(FloatType())
def get_max_job_sim(sim_matrix):
    """
    Get Max Job Sim
    """
    if len(sim_matrix) > 0:
        return max(sim_matrix)
    else:
        return 0
    
@udf(BooleanType())
def is_freshie(sim_matrix):
    """
    Boolean to determine if the person is new to the job market
    """
    return len(sim_matrix) == 0

###################################################
# Individual silver tables processing
###################################################
def data_processing_silver_skills_ref(spark: SparkSession):
    pass

def data_processing_silver_education_ref(spark: SparkSession):
    pass

def data_processing_silver_resume(snapshot_date : datetime, type, spark: SparkSession):
    """
    Processes resumes from bronze layer to silver layer
    Output: saves into parquet
    """    
    # Read from S3
    filename = f"{snapshot_date.year}-{snapshot_date.month:02d}.parquet"
    # s3_key = f"datamart/online/bronze/resume/{filename}"
    # df = read_s3_data('jobmirror-s3', s3_key, spark)
    if type == "training":
        filepath =  f"datamart/bronze/resume/{filename}"
    elif type == "inference":
        filepath = f"datamart/online/bronze/resume/{filename}"
    
    df = read_bronze_resume_as_pyspark(snapshot_date, type, spark)

    # Add skills columns
    df = create_hard_skills_column(df, spark, og_column="hard_skills")
    df = create_soft_skills_column(df, spark, og_column="soft_skills")

    # Add education columns - S3 reads
    # edu_levels = spark.sparkContext.broadcast(
    #     read_s3_data('jobmirror-s3', "datamart/references/education_level_synonyms.parquet", spark).collect()
    # )
    # edu_fields = spark.sparkContext.broadcast(
    #     read_s3_data('jobmirror-s3', "datamart/references/education_field_synonyms.parquet", spark).collect()
    # )
    # cert_categories = spark.sparkContext.broadcast(
    #     read_s3_data('jobmirror-s3', "datamart/references/certification_categories.parquet", spark).collect()
    # )
    
    # Add education columns
    # education_ref_dir = os.path.join("datamart", "references")
    # edu_levels = spark.sparkContext.broadcast(spark.read.parquet(os.path.join(education_ref_dir, "education_level_synonyms.parquet")).collect())
    # edu_fields = spark.sparkContext.broadcast(spark.read.parquet(os.path.join(education_ref_dir, "education_field_synonyms.parquet")).collect())
    # cert_categories = spark.sparkContext.broadcast(spark.read.parquet(os.path.join(education_ref_dir, "certification_categories.parquet")).collect())
    
    project_root = Path("/opt/airflow")
    education_ref_dir = project_root / "datamart/references"

    # Build full paths to reference Parquet files
    edu_levels_path = education_ref_dir / "education_level_synonyms.parquet"
    edu_fields_path = education_ref_dir / "education_field_synonyms.parquet"
    cert_categories_path = education_ref_dir / "certification_categories.parquet"

    # Broadcast them
    edu_levels = spark.sparkContext.broadcast(spark.read.parquet(str(edu_levels_path)).collect())
    edu_fields = spark.sparkContext.broadcast(spark.read.parquet(str(edu_fields_path)).collect())
    cert_categories = spark.sparkContext.broadcast(spark.read.parquet(str(cert_categories_path)).collect())

    _edu_udf = parse_education_udf_factory()
    df = (
        df
        .withColumn("tmp_edu", _edu_udf("education"))
        .withColumn("edu_highest_level", udf(lambda x: determine_edu_mapping(x, edu_levels.value, 30), StringType())("tmp_edu.edu_desc"))
        .withColumn("edu_field", udf(lambda x: determine_edu_mapping(x, edu_fields.value, 70), StringType())("tmp_edu.edu_desc"))
        .withColumn("edu_gpa", col("tmp_edu.edu_gpa"))
        .withColumn("edu_institution", col("tmp_edu.edu_institution"))
        .withColumn("cert_categories", udf(lambda x: determine_certification_types(x, cert_categories.value, 80), ArrayType(StringType()))("certifications"))
        .drop("tmp_edu")
        .drop("education")
        .drop("certifications")
    )

    # Add experience columns
    df = df.withColumn("YoE_list", get_resume_yoe("experience"))

    # Add miscellaneous columns
    df = clean_employment_type_column(df, "employment_type_preference") \
        .drop("employment_type_preference") \
        .withColumnRenamed("employment_type_preference_cleaned", "employment_type_preference")
    df = clean_work_authorization_column(df, "work_authorization") \
        .drop("work_authorization") \
        .withColumnRenamed("work_authorization_cleaned", "work_authorization")
    location_lookup_dict = location_lookup()
    df = standardize_location_column(df, "location_preference", location_lookup_dict) \
        .drop("location_preference") \
        .withColumnRenamed("location_preference_cleaned", "location_preference")
    
    # Write directly to S3
    # if type == "training":
    #     s3_output_key = f"datamart/silver/resumes/{output_filename}"
    # elif type == "inference":
    #     s3_output_key = f"datamart/silver/online/resumes/{output_filename}"
    
    # s3_output_path = f"s3a://jobmirror-s3/{s3_output_key}"
    # df.write.mode("overwrite").parquet(s3_output_path)

    project_root = Path("/opt/airflow") 
    output_filename = f"{snapshot_date.year}-{snapshot_date.month}.parquet"
    # Determine local output path based on type
    if type == "training":
        local_output_path = project_root / "datamart/silver/resumes" / output_filename
    elif type == "inference":
        local_output_path = project_root / "datamart/silver/online/resumes" / output_filename

    # Ensure the output directory exists
    local_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the dataframe locally
    df.write.mode("overwrite").parquet(str(local_output_path))
    
    print(f"Saved Silver Resume to: {output_filename} No. Rows: {df.count()}")


def data_processing_silver_jd(snapshot_date : datetime, type, spark: SparkSession):
    """
    Processes job descriptions from bronze layer to silver layer
    Output: saves into parquet
    """

    # Read into pyspark dataframe
    df = read_bronze_jd_as_pyspark(snapshot_date, type, spark)

    df = df.withColumnRenamed("certifications", "jd_certifications")

    # Add skills columns
    df = create_hard_skills_column(df, spark, og_column="required_hard_skills")
    df = create_soft_skills_column(df, spark, og_column="required_soft_skills")

    # Add education columns - S3 reads
    # edu_levels = spark.sparkContext.broadcast(read_s3_data('jobmirror-s3', "datamart/references/education_level_synonyms.parquet", spark).collect())
    # edu_fields = spark.sparkContext.broadcast(read_s3_data('jobmirror-s3', "datamart/references/education_field_synonyms.parquet", spark).collect())
    # cert_categories = spark.sparkContext.broadcast(read_s3_data('jobmirror-s3', "datamart/references/certification_categories.parquet", spark).collect())

    # Add education columns
    project_root = Path("/opt/airflow")
    education_ref_dir = project_root / "datamart/references"

    # Build full paths to reference Parquet files
    edu_levels_path = education_ref_dir / "education_level_synonyms.parquet"
    edu_fields_path = education_ref_dir / "education_field_synonyms.parquet"
    cert_categories_path = education_ref_dir / "certification_categories.parquet"

    # Broadcast them
    edu_levels = spark.sparkContext.broadcast(spark.read.parquet(str(edu_levels_path)).collect())
    edu_fields = spark.sparkContext.broadcast(spark.read.parquet(str(edu_fields_path)).collect())
    cert_categories = spark.sparkContext.broadcast(spark.read.parquet(str(cert_categories_path)).collect())
    
    df = (
        df
        .withColumn("required_edu_level", udf(lambda x: determine_edu_mapping(x, edu_levels.value, 30), StringType())("required_education"))
        .withColumn("required_edu_field", udf(lambda x: determine_edu_mapping(x, edu_fields.value, 70), StringType())("required_education"))
        .withColumn("required_cert_categories", udf(lambda x: determine_certification_types(x, cert_categories.value, 80), ArrayType(StringType()))("jd_certifications"))
        .drop("required_education")
        .drop("jd_certifications")
    )

    # Add miscellaneous columns
    df = clean_employment_type_column(df, "employment_type") \
        .drop("employment_type") \
        .withColumnRenamed("employment_type_cleaned", "employment_type")
    df = clean_work_authorization_column(df, "required_work_authorization") \
        .drop("required_work_authorization") \
        .withColumnRenamed("required_work_authorization_cleaned", "required_work_authorization")
    location_lookup_dict = location_lookup()
    df = standardize_location_column(df, "job_location", location_lookup_dict) \
        .drop("job_location") \
        .withColumnRenamed("job_location_cleaned", "job_location")
    
    
    filename = f"{snapshot_date.year}-{snapshot_date.month}.parquet"

    # Write directly to S3
    # if type == "training":
    #     s3_output_key = f"datamart/silver/job_descriptions/{filename}"
    # elif type == "inference":
    #     s3_output_key = f"datamart/silver/online/job_descriptions/{filename}"

    # s3_output_path = f"s3a://jobmirror-s3/{s3_output_key}"
    # df.write.mode("overwrite").parquet(s3_output_path)

    project_root = Path("/opt/airflow") 
    # Determine local output path
    if type == "training":
        local_output_path = project_root / "datamart/silver/job_descriptions" / filename
    elif type == "inference":
        local_output_path = project_root / "datamart/silver/online/job_descriptions" / filename

    # Ensure the directory exists
    local_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the DataFrame locally
    df.write.mode("overwrite").parquet(str(local_output_path))
    
    print(f"Saved Silver JD to: {filename} No. Rows: {df.count()}")


def data_processing_silver_labels(snapshot_date : datetime, type, spark: SparkSession):
    """
    Processes labels from bronze layer to silver layer
    Output: saves into parquet
    """

    # Read into pyspark dataframe
    df = read_bronze_labels_as_pyspark(snapshot_date, type, spark)

    # Group labels together
    df = standardize_label(df, "fit") \
        .select('resume_id', 'job_id', 'snapshot_date', 'fit_cleaned') \
        .withColumnRenamed('fit_cleaned', 'fit')
    
    # Write directly to S3
    selected_date = str(snapshot_date.year) + "-" + str(snapshot_date.month)

    # filename = selected_date + ".parquet"
    # if type == "training":
    #     s3_output_key = f"datamart/silver/labels/{filename}"
    # elif type == "inference":
    #     s3_output_key = f"datamart/silver/online/labels/{filename}"

    # s3_output_path = f"s3a://jobmirror-s3/{s3_output_key}"
    # df.write.mode("overwrite").parquet(s3_output_path)

    project_root = Path("/opt/airflow")
    selected_date = f"{snapshot_date.year}-{snapshot_date.month:02}"
    filename = f"{selected_date}.parquet"

    # Determine local output path
    if type == "training":
        local_output_path = project_root / "datamart/silver/labels" / filename
    elif type == "inference":
        local_output_path = project_root / "datamart/silver/online/labels" / filename

    # Ensure the output directory exists
    local_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the DataFrame to local filesystem
    df.write.mode("overwrite").parquet(str(local_output_path))

    print(f"Saved Silver Labels to: {selected_date} No. Rows: {df.count()}")
    
def data_processing_silver_combined(snapshot_date: datetime, type, spark : SparkSession) -> None:
    """
    Merge the parquets together, get the dataframe for further processing
    """

    selected_date = str(snapshot_date.year) + "-" + str(snapshot_date.month)
    filename = f"{selected_date}.parquet"
    # if type == "training":
    #     jd_key = f"datamart/silver/job_descriptions/{selected_date}.parquet"
    #     resume_key = f"datamart/silver/resumes/{selected_date}.parquet"
    #     labels_key = f"datamart/silver/labels/{selected_date}.parquet"
    # elif type == "inference":
    #     jd_key = f"datamart/silver/online/job_descriptions/{selected_date}.parquet"
    #     resume_key = f"datamart/silver/online/resumes/{selected_date}.parquet"
    #     labels_key = f"datamart/silver/online/labels/{selected_date}.parquet"
    
    # # Read from S3
    # jd_df = read_s3_data('jobmirror-s3', jd_key, spark)
    # resume_df = read_s3_data('jobmirror-s3', resume_key, spark)
    # labels_df = read_s3_data('jobmirror-s3', labels_key, spark)

    project_root = Path("/opt/airflow")

    # Determine file paths based on type
    if type == "training":
        jd_path = project_root / "datamart/silver/job_descriptions" / filename
        resume_path = project_root / "datamart/silver/resumes" / filename
        labels_path = project_root / "datamart/silver/labels" / filename
    elif type == "inference":
        jd_path = project_root / "datamart/silver/online/job_descriptions" / filename
        resume_path = project_root / "datamart/silver/online/resumes" / filename
        labels_path = project_root / "datamart/silver/online/labels" / filename

    # Read from local file system
    jd_df = spark.read.parquet(str(jd_path))
    resume_df = spark.read.parquet(str(resume_path))
    labels_df = spark.read.parquet(str(labels_path))

    # We do the individual transforms to dfs first
    ## Resume Transforms
    resume_df = resume_df.withColumnRenamed("id", "resume_id")
    resume_df = resume_df.withColumnRenamed("snapshot_date", "resume_snapshot")
  
    ## JD Transforms
    jd_df = jd_df.withColumnRenamed("id", "job_id")
    jd_df = jd_df.withColumnRenamed("snapshot_date", "job_snapshot")
    jd_df = jd_df.withColumnRenamed("hard_skills_general", "jd_hard_skills_general") \
            .withColumnRenamed("hard_skills_specific", "jd_hard_skills_specific") \
            .withColumnRenamed("soft_skills", "jd_soft_skills")
    
    # Combine the parquets together
    labels_jd = labels_df.join(jd_df, on="job_id", how="inner")
    combined_jd_resume = labels_jd.join(resume_df, on="resume_id", how="inner")

    """
    DO UDFS HERE
    """
    # Get the experience similarity score 
    combined_jd_resume = combined_jd_resume.withColumn("exp_sim_list", get_title_similarity_score(combined_jd_resume['role_title'], combined_jd_resume['experience']))

    project_root = Path("/opt/airflow")

    # Create filename with UUID suffix
    filename = f"{selected_date}{str(uuid.uuid4())[:8]}.parquet"

    # Determine local output path
    if type == "training":
        local_output_path = project_root / "datamart/silver/combined_jd_resume" / filename
    elif type == "inference":
        local_output_path = project_root / "datamart/silver/online/combined_jd_resume" / filename

    # Ensure the output directory exists
    local_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the combined DataFrame to local disk
    combined_jd_resume.write.mode("overwrite").parquet(str(local_output_path))

    # Write directly to S3
    # if type == "training":
    #     s3_output_key = f"datamart/silver/combined_resume_jd/{filename}"
    # elif type == "inference":
    #     s3_output_key = f"datamart/silver/online/combined_resume_jd/{filename}"

    # s3_output_path = f"s3a://jobmirror-s3/{s3_output_key}"
    # combined_jd_resume.write.mode("overwrite").parquet(s3_output_path)

    # print(f"Saved Silver Combined: {selected_date} No. Rows: {combined_jd_resume.count()}")

if __name__ == "__main__":

    # Get the pyspark session
    spark = get_pyspark_session()

    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--task", type=str, required=True, help="Which task to run")
    parser.add_argument('--type', type=str, default='training', help='Inference or training')
    
    args = parser.parse_args()

    # Convert string to datetime object
    snapshot_date = datetime.strptime(args.snapshotdate, "%Y-%m-%d")    

    if args.task == "data_processing_silver_resume":
        data_processing_silver_resume(snapshot_date, args.type, spark)
    elif args.task == "data_processing_silver_jd":
        data_processing_silver_jd(snapshot_date, args.type, spark)
    elif args.task == "data_processing_silver_combined":
        data_processing_silver_combined(snapshot_date, args.type, spark)
    elif args.task == "data_processing_silver_labels":
        data_processing_silver_labels(snapshot_date, args.type, spark)
    else:
        raise ValueError(f"Unknown task: {args.task}")