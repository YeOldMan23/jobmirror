"""
Read the parquet files and extract new information out of them
"""

from .mongodb_utils import read_bronze_resume_as_pyspark, read_bronze_jd_as_pyspark, read_bronze_labels_as_pyspark
from .gdrive_utils import connect_to_gdrive, get_folder_id_by_path, upload_file_to_drive

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

import os

import torch
from datetime import datetime

###################################################
# Individual silver tables processing
###################################################
def data_processing_silver_skills_ref(spark: SparkSession):
    pass

def data_processing_silver_education_ref(spark: SparkSession):
    pass

def data_processing_silver_resume(snapshot_date : datetime, spark: SparkSession):
    """
    Processes resumes from bronze layer to silver layer
    Output: saves into parquet
    """

    service = connect_to_gdrive()
        
    parent_root = '1_eMgnRaFtt-ZSZD3zfwai3qlpYJ-M5C6' 
        
    resume_path = ['datamart', 'silver', 'resume']
    resume_id = get_folder_id_by_path(service, resume_path, parent_root)
    print("\nResume folder ID:", resume_id)
    
    # Read into pyspark dataframe
    df = read_bronze_resume_as_pyspark(snapshot_date, spark)

    # Add skills columns
    df = create_hard_skills_column(df, spark, og_column="hard_skills")
    df = create_soft_skills_column(df, spark, og_column="soft_skills")

    # Add education columns
    education_ref_dir = os.path.join("datamart", "references")
    edu_levels = spark.sparkContext.broadcast(spark.read.parquet(os.path.join(education_ref_dir, "education_level_synonyms.parquet")).collect())
    edu_fields = spark.sparkContext.broadcast(spark.read.parquet(os.path.join(education_ref_dir, "education_field_synonyms.parquet")).collect())
    cert_categories = spark.sparkContext.broadcast(spark.read.parquet(os.path.join(education_ref_dir, "certification_categories.parquet")).collect())

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

    # Save table as parquet
    filename = str(snapshot_date.year) + "-" + str(snapshot_date.month) + ".parquet"
    output_path = os.path.join("datamart", "silver", "resumes", filename)
    df.write.mode("overwrite").parquet(output_path)

    # upload_file_to_drive(service, output_path, resume_id)


def data_processing_silver_jd(snapshot_date : datetime, spark: SparkSession):
    """
    Processes job descriptions from bronze layer to silver layer
    Output: saves into parquet
    """

    service = connect_to_gdrive()
        
    parent_root = '1_eMgnRaFtt-ZSZD3zfwai3qlpYJ-M5C6' 
        
        
    jd_path = ['datamart', 'silver',  'job_description']
    jd_id = get_folder_id_by_path(service, jd_path, parent_root)
    print("\nJob description folder ID:", jd_id)

    # Read into pyspark dataframe
    df = read_bronze_jd_as_pyspark(snapshot_date, spark)
    df = df.withColumnRenamed("certifications", "jd_certifications")

    # Add skills columns
    df = create_hard_skills_column(df, spark, og_column="required_hard_skills")
    df = create_soft_skills_column(df, spark, og_column="required_soft_skills")

    # Add education columns
    education_ref_dir = os.path.join("datamart", "references")
    edu_levels = spark.sparkContext.broadcast(spark.read.parquet(os.path.join(education_ref_dir, "education_level_synonyms.parquet")).collect())
    edu_fields = spark.sparkContext.broadcast(spark.read.parquet(os.path.join(education_ref_dir, "education_field_synonyms.parquet")).collect())
    cert_categories = spark.sparkContext.broadcast(spark.read.parquet(os.path.join(education_ref_dir, "certification_categories.parquet")).collect())

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

    # Save table as parquet
    filename = str(snapshot_date.year) + "-" + str(snapshot_date.month) + ".parquet"
    output_path = os.path.join("datamart", "silver", "job_descriptions", filename)
    df.write.mode("overwrite").parquet(output_path)

    upload_file_to_drive(service, output_path, jd_id)

def data_processing_silver_labels(snapshot_date : datetime, spark: SparkSession):
    """
    Processes labels from bronze layer to silver layer
    Output: saves into parquet
    """

    service = connect_to_gdrive()
        
    parent_root = '1_eMgnRaFtt-ZSZD3zfwai3qlpYJ-M5C6' 
        
        
    jd_path = ['datamart', 'silver',  'label']
    label_id = get_folder_id_by_path(service, jd_path, parent_root)
    print("\nLabel folder ID:", label_id)

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

    print(f"Saved Silver Labels : {selected_date} No. Rows : {df.count()}")

    upload_file_to_drive(service, output_path, label_id)
    

def data_processing_silver_combined(snapshot_date: datetime, spark : SparkSession) -> None:
    """
    Merge the parquets together, get the dataframe for further processing
    """

    service = connect_to_gdrive()
        
    parent_root = '1_eMgnRaFtt-ZSZD3zfwai3qlpYJ-M5C6' 
        
        
    combine_path = ['datamart', 'silver',  'combined_resume_jd']
    combined_id = get_folder_id_by_path(service, combine_path, parent_root)
    print("\nCombined folder ID:", combined_id)

    selected_date = str(snapshot_date.year) + "-" + str(snapshot_date.month)
    jd_full_dir     = os.path.join("datamart", "silver", "job_descriptions", f"{selected_date}.parquet")
    resume_full_dir = os.path.join("datamart", "silver", "resumes", f"{selected_date}.parquet")
    labels_full_dir = os.path.join("datamart", "silver", "labels", f"{selected_date}.parquet")

    jd_df     = spark.read.parquet(jd_full_dir)
    resume_df = spark.read.parquet(resume_full_dir)
    labels_df = spark.read.parquet(labels_full_dir)

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
    labels_jd_resume = labels_jd.join(resume_df, on="resume_id", how="inner")

    """
    DO UDFS HERE
    """
    # Get the experience similarity score 
    labels_jd_resume = labels_jd_resume.withColumn("exp_sim_list", get_title_similarity_score(labels_jd_resume['role_title'], labels_jd_resume['experience']))

    # Save the parquet 
    filename    = selected_date + ".parquet"
    output_path = os.path.join("datamart", "silver", "combined", filename)
    labels_jd_resume.write.mode("overwrite").parquet(output_path)

    print(f"Saved Silver Combined : {selected_date} No. Rows : {labels_jd_resume.count()}")

    upload_file_to_drive(service, output_path, combined_id)