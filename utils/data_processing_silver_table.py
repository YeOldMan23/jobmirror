"""
Read the parquet files and extract new information out of them
"""

from .mongodb_utils import read_bronze_resume_as_pyspark, read_bronze_jd_as_pyspark, read_bronze_labels_as_pyspark

# Get the features to match
from .silver_feature_extraction.extract_features_jd import *
from .silver_feature_extraction.extract_features_resume import *
from .silver_feature_extraction.match_features import *
from .silver_feature_extraction.extract_exp import get_resume_yoe, get_title_similarity_score
from .silver_feature_extraction.extract_edu import parse_education_udf_factory, determine_edu_mapping
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

def data_processing_silver_resume(snapshot_date : datetime, spark: SparkSession):
    """
    Processes resumes from bronze layer to silver layer
    Output: saves into parquet
    """
    # Read into pyspark dataframe
    df = read_bronze_resume_as_pyspark(snapshot_date, spark)

    # Add education columns
    _edu_udf = parse_education_udf_factory()
    df = (
        df
        .withColumn("edu_tmp", _edu_udf("education"))
        .withColumn("highest_level_education", col("edu_tmp.highest_level_education"))
        .withColumn("major",                   col("edu_tmp.major"))
        .withColumn("gpa",                     col("edu_tmp.gpa"))
        .withColumn("institution",             col("edu_tmp.institution"))
        .drop("edu_tmp")
    )

    # Add experience columns
    df = df.withColumn("YoE_list", get_resume_yoe("experience"))

    # Add skills columns
    df = create_hard_skills_column(df, spark, og_column="hard_skills")
    df = create_soft_skills_column(df, spark, og_column="soft_skills")

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

def data_processing_silver_jd(snapshot_date : datetime, spark: SparkSession):
    """
    Processes job descriptions from bronze layer to silver layer
    Output: saves into parquet
    """
    # Read into pyspark dataframe
    df = read_bronze_jd_as_pyspark(snapshot_date, spark)
    df = df.withColumnRenamed("certifications", "jd_certifications")

    # Add education columns
    education_ref_dir = os.path.join("datamart", "references")
    edu_levels = spark.sparkContext.broadcast(spark.read.parquet(os.path.join(education_ref_dir, "education_level_synonyms.parquet")).collect())
    edu_fields = spark.sparkContext.broadcast(spark.read.parquet(os.path.join(education_ref_dir, "education_field_synonyms.parquet")).collect())
    cert_categories = spark.sparkContext.broadcast(spark.read.parquet(os.path.join(education_ref_dir, "certification_categories.parquet")).collect())

    df = (
        df
        .withColumn("required_edu_level", udf(lambda x: determine_edu_mapping(x, edu_levels.value), StringType())("required_education"))
        .withColumn("required_edu_field", udf(lambda x: determine_edu_mapping(x, edu_fields.value), StringType())("required_education"))
        .withColumn("required_cert_field", udf(lambda x: determine_edu_mapping(x, cert_categories.value), StringType())("jd_certifications"))
        .withColumn("no_of_certs", udf(lambda x: len(x) if isinstance(x, list) else 0, IntegerType())("jd_certifications"))
        .drop("required_education")
        .drop("jd_certifications")
    )

    # Add skills columns
    df = create_hard_skills_column(df, spark, og_column="required_hard_skills")
    df = create_soft_skills_column(df, spark, og_column="required_soft_skills")

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

def data_processing_silver_labels(snapshot_date : datetime, spark: SparkSession):
    """
    Processes labels from bronze layer to silver layer
    Output: saves into parquet
    """
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
    

def data_processing_silver_combined(snapshot_date: datetime, spark : SparkSession) -> None:
    """
    Merge the parquets together, get the dataframe for further processing
    """
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