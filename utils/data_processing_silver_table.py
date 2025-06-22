"""
Read the parquet files and extract new information out of them
"""

from .mongodb_utils import read_bronze_resume_as_pyspark, read_bronze_jd_as_pyspark, read_bronze_labels_as_pyspark
from .gdrive_utils import connect_to_gdrive, get_folder_id_by_path, upload_file_to_drive
from .s3_utils import upload_to_s3, read_parquet_from_s3
from .config import AWSConfig

# Get the features to match
from .silver_feature_extraction.extract_exp import get_resume_yoe, get_title_similarity_score
from .silver_feature_extraction.extract_edu import parse_education_udf_factory, determine_edu_mapping, determine_certification_types
from .silver_feature_extraction.extract_skills import create_hard_skills_column, create_soft_skills_column
from .silver_feature_extraction.extract_misc import clean_employment_type_column, location_lookup, standardize_location_column, clean_work_authorization_column, standardize_label

# Get Spark Session
import pyspark
from pyspark.sql.dataframe import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import FloatType, StructType, StructField, StringType, ArrayType, BooleanType, IntegerType
from utils.mongodb_utils import get_pyspark_session

import os
import shutil
import boto3
import argparse

import torch
from datetime import datetime
from typing import Optional

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

def data_processing_silver_resume(snapshot_date : datetime, spark: SparkSession, type):
    """
    Processes resumes from bronze layer to silver layer
    Output: saves into parquet
    """
    # commenting out to test S3

    # service = connect_to_gdrive()
        
    # parent_root = '1_eMgnRaFtt-ZSZD3zfwai3qlpYJ-M5C6' 
        
    # resume_path = ['datamart', 'silver', 'resume']
    # resume_id = get_folder_id_by_path(service, resume_path, parent_root)
    # print("\nResume folder ID:", resume_id)
    
    # Read into pyspark dataframe
    if type == "training": 
        df = read_bronze_resume_as_pyspark(snapshot_date, spark)
    elif type == "inference":
        filename = f"{snapshot_date.year}-{snapshot_date.month:02d}.parquet"
        s3_key = f"datamart/online/bronze/resume/{filename}"
        df = read_parquet_from_s3(spark,s3_key) 
    
    # Add skills columns
    df = create_hard_skills_column(df, spark, og_column="hard_skills")
    df = create_soft_skills_column(df, spark, og_column="soft_skills")

    # Add education columns
    education_ref_dir = os.path.join("datamart", "references")
    edu_levels = spark.sparkContext.broadcast(spark.read.parquet(os.path.join(education_ref_dir, "education_level_synonyms.parquet")).collect())
    edu_fields = spark.sparkContext.broadcast(spark.read.parquet(os.path.join(education_ref_dir, "education_field_synonyms.parquet")).collect())
    cert_categories = spark.sparkContext.broadcast(spark.read.parquet(os.path.join(education_ref_dir, "certification_categories.parquet")).collect())

    # Change to S3 reads:
    # edu_levels = spark.sparkContext.broadcast(
    #     read_parquet_from_s3(spark, "datamart/references/education_level_synonyms.parquet").collect()
    # )
    # edu_fields = spark.sparkContext.broadcast(
    #     read_parquet_from_s3(spark, "datamart/references/education_field_synonyms.parquet").collect()
    # )
    # cert_categories = spark.sparkContext.broadcast(
    #     read_parquet_from_s3(spark, "datamart/references/certification_categories.parquet").collect()
    # )

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

    if type == "training": 
        # Save table as parquet
        filename = str(snapshot_date.year) + "-" + str(snapshot_date.month) + ".parquet"
        output_path = os.path.join("datamart", "silver", "online","resumes", filename)
        df.write.mode("overwrite").parquet(output_path)
        
        # uploading to s3
        s3_key = f"datamart/silver/resume/{filename}"
        upload_to_s3(output_path, s3_key)
        
    elif type == "inference":
        print(df)
        return df
        # Save table as parquet
        # filename = str(snapshot_date.year) + "-" + str(snapshot_date.month) + ".parquet"
        # output_path = os.path.join("datamart","online", "silver", "resumes", filename)
        # df.write.mode("overwrite").parquet(output_path)
        
        # # uploading to s3
        # s3_key = f"datamart/online/silver/resumes/{filename}"
        # upload_to_s3(output_path, s3_key)        
    
    # path is for temp storage of files. this function is to clear the folder when complete.
    # shutil.rmtree(local_path)
    
    # upload_file_to_drive(service, output_path, resume_id)


def data_processing_silver_jd(snapshot_date : datetime, spark: SparkSession, type):
    """
    Processes job descriptions from bronze layer to silver layer
    Output: saves into parquet
    """
    # service = connect_to_gdrive()
        
    # parent_root = '1_eMgnRaFtt-ZSZD3zfwai3qlpYJ-M5C6' 
                
    jd_path = ['datamart', 'silver',  'job_description']
    # jd_id = get_folder_id_by_path(service, jd_path, parent_root)
    # print("\nJob description folder ID:", jd_id)

    # Read into pyspark dataframe
    # df = read_bronze_jd_as_pyspark(snapshot_date, spark)
        # Read into pyspark dataframe

    if type == "training": 
        df = read_bronze_jd_as_pyspark(snapshot_date, spark)
    elif type == "inference":
        filename = f"{snapshot_date.year}-{snapshot_date.month:02d}.parquet"
        s3_key = f"datamart/online/bronze/jd/{filename}"
        df = read_parquet_from_s3(spark,s3_key)

    df = df.withColumnRenamed("certifications", "jd_certifications")

    # Add skills columns
    df = create_hard_skills_column(df, spark, og_column="required_hard_skills")
    df = create_soft_skills_column(df, spark, og_column="required_soft_skills")

    # Add education columns
    education_ref_dir = os.path.join("datamart", "references")
    edu_levels = spark.sparkContext.broadcast(spark.read.parquet(os.path.join(education_ref_dir, "education_level_synonyms.parquet")).collect())
    edu_fields = spark.sparkContext.broadcast(spark.read.parquet(os.path.join(education_ref_dir, "education_field_synonyms.parquet")).collect())
    cert_categories = spark.sparkContext.broadcast(spark.read.parquet(os.path.join(education_ref_dir, "certification_categories.parquet")).collect())
    
    # edu_levels = spark.sparkContext.broadcast(
    #     read_parquet_from_s3(spark, "datamart/references/education_level_synonyms.parquet").collect()
    # )
    # edu_fields = spark.sparkContext.broadcast(
    #     read_parquet_from_s3(spark, "datamart/references/education_field_synonyms.parquet").collect()
    # )
    # cert_categories = spark.sparkContext.broadcast(
    #     read_parquet_from_s3(spark, "datamart/references/certification_categories.parquet").collect()
    # )    

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
    
    if type == "training":

        # Save table as parquet
        filename = str(snapshot_date.year) + "-" + str(snapshot_date.month) + ".parquet"
        output_path = os.path.join("datamart", "silver", "job_descriptions", filename)
        df.write.mode("overwrite").parquet(output_path)

        # uploading to s3
        s3_key = f"datamart/silver/job_description/{filename}"
        upload_to_s3(output_path, s3_key)

    elif type == "inference":
        print(df)
        return df

    # upload_file_to_drive(service, output_path, jd_id)

def data_processing_silver_labels(snapshot_date : datetime, spark: SparkSession):
    """
    Processes labels from bronze layer to silver layer
    Output: saves into parquet
    """

    # service = connect_to_gdrive()
        
    # parent_root = '1_eMgnRaFtt-ZSZD3zfwai3qlpYJ-M5C6' 
        
    # jd_path = ['datamart', 'silver',  'label']
    # label_id = get_folder_id_by_path(service, jd_path, parent_root)
    # print("\nLabel folder ID:", label_id)

    # Read into pyspark dataframe
    df = read_bronze_labels_as_pyspark(snapshot_date, spark)

    # Group labels together
    df = standardize_label(df, "fit") \
        .select('resume_id', 'job_id', 'snapshot_date', 'fit_cleaned') \
        .withColumnRenamed('fit_cleaned', 'fit')

    # Save table as parquet
    selected_date = str(snapshot_date.year) + "-" + str(snapshot_date.month)
    filename = selected_date + ".parquet"
    output_path = os.path.join("datamart", "silver", "labels", filename)
    df.write.mode("overwrite").parquet(output_path)

    # uploading to s3
    s3_key = f"datamart/silver/labels/{filename}"
    upload_to_s3(output_path, s3_key)

    print(f"Saved Silver Labels : {selected_date} No. Rows : {df.count()}")

    # upload_file_to_drive(service, output_path, label_id)
    
def data_processing_silver_combined(snapshot_date: datetime,
                                     spark: SparkSession,
                                     type: str,
                                     resume_df: Optional[DataFrame] = None,
                                     jd_df: Optional[DataFrame] = None) -> Optional[DataFrame]:
# def data_processing_silver_combined(snapshot_date: datetime, spark : SparkSession) -> None:
    """
    Merge the parquets together, get the dataframe for further processing
    """

    # service = connect_to_gdrive()
        
    # parent_root = '1_eMgnRaFtt-ZSZD3zfwai3qlpYJ-M5C6' 
                
    # combine_path = ['datamart', 'silver',  'combined_resume_jd']
    # combined_id = get_folder_id_by_path(service, combine_path, parent_root)
    # print("\nCombined folder ID:", combined_id)
    if type == "training":
        selected_date = str(snapshot_date.year) + "-" + str(snapshot_date.month)
        jd_full_dir     = os.path.join("datamart", "silver", "job_descriptions", f"{selected_date}.parquet")
        resume_full_dir = os.path.join("datamart", "silver", "resumes", f"{selected_date}.parquet")
        labels_full_dir = os.path.join("datamart", "silver", "labels", f"{selected_date}.parquet")

        jd_df     = spark.read.parquet(jd_full_dir)
        resume_df = spark.read.parquet(resume_full_dir)
        labels_df = spark.read.parquet(labels_full_dir)

    elif type == "inference":
         if resume_df is None or jd_df is None:
            raise ValueError("resume_df and jd_df must be provided in inference mode")

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

    if type == "training":
        filename    = selected_date + ".parquet"
        output_path = os.path.join("datamart", "silver", "combined_resume_jd", filename)
        combined_jd_resume.write.mode("overwrite").parquet(output_path)
        print(f"Saved Silver Combined : {selected_date} No. Rows : {combined_jd_resume.count()}")
    elif type == "inference":
        return combined_jd_resume

    # # Split into training and inference data based on label availability
    # training_data = labels_jd_resume.filter(col("fit").isNotNull())  # Has labels
    # inference_data = labels_jd_resume.filter(col("fit").isNull())    # No labels

    # # Save training data
    # if training_data.count() > 0:
    #     training_output_path = os.path.join("datamart", "silver", "combined_resume_jd", f"{selected_date}.parquet")
    #     training_data.write.mode("overwrite").parquet(training_output_path)
    #     print(f"Saved {training_data.count()} training records")
    #     print(f"Saved Silver Combined : {selected_date} No. Rows : {labels_jd_resume.count()}")

    # # Save inference data  
    # if inference_data.count() > 0:
    #     # Drop the null label column for inference
    #     inference_clean = inference_data.drop("fit")
    #     inference_output_path = os.path.join("datamart", "silver", "combined_inference", f"{selected_date}.parquet")
    #     inference_clean.write.mode("overwrite").parquet(inference_output_path)
    #     print(f"Saved {inference_data.count()} inference records")

    # # Save the parquet 
    # filename    = selected_date + ".parquet"
    # output_path = os.path.join("datamart", "silver", "combined_resume_jd", filename)
    # labels_jd_resume.write.mode("overwrite").parquet(output_path)
    # print(f"Saved Silver Combined : {selected_date} No. Rows : {labels_jd_resume.count()}")

    # upload_file_to_drive(service, output_path, combined_id)

if __name__ == "__main__":

    spark = get_pyspark_session()

    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--task", type=str, required=True, help="Which task to run")
    parser.add_argument('--type', type=str, default='training', help='Inference or training')
    
    args = parser.parse_args()

    if args.task == "data_processing_silver_resume":
        data_processing_silver_resume(args.snapshotdate)
    elif args.task == "data_processing_silver_jd":
        data_processing_silver_jd(args.snapshotdate,args.type)
    elif args.task == "data_processing_silver_combined":
        data_processing_silver_combined(args.snapshotdate)
    elif args.task == "data_processing_silver_labels":
        data_processing_silver_labels(args.snapshotdate)

    else:
        raise ValueError(f"Unknown task: {args.task}")