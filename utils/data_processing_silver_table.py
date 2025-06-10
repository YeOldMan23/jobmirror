"""
Read the parquet files and extract new information out of them
"""

# Get the features to match
from .feature_extraction.extract_features_jd import *
from .feature_extraction.extract_features_resume import *
from .feature_extraction.match_features import *

# Get Spark Session
import pyspark
from pyspark.sql.dataframe import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

# Using another language model to read the values
from sentence_transformers import SentenceTransformer, util

import os

"""
Global Variables
"""
device = None
if torch.cuda.is_available():
    print("Using GPU")
    device = torch.cuda.get_device_name(0)

embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5", device = device)

"""
Custom Functions for getting values
"""

def get_time_difference(start_time : datetime, end_time : datetime) -> float:
    """
    GGet the time difference between the start time and end time
    return as a float of years
    """
    time_diff = (end_time - start_time).days / 365.25

    return time_diff

def parse_flexible_date(date_str) -> datetime:
    try:
        # Try full ISO first
        return datetime.fromisoformat(date_str)
    except (ValueError, TypeError):
        pass

    # If None, just return last date of 2021
    if not date_str:
        return datetime(2021, 12, 31)

    # Match YYYY
    if re.fullmatch(r"\d{4}", date_str):
        return datetime(int(date_str), 12, 31)
    
    # Match YYYY-MM
    if re.fullmatch(r"\d{4}-\d{2}", date_str):
        year, month = map(int, date_str.split("-"))
        return datetime(year, month, 28) # Safest day to take
    
    # Match YYYY-MM-DD
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_str):
        return datetime.strptime(date_str, "%Y-%m-%d")
    
    # Else just return the last date of 2021 , date is either current/present
    return datetime(2021, 12, 31)

@udf(FloatType())
def get_resume_yoe(experience_array) -> float:
    """
    Get the YoE from resume experiences
    """
    total_yoe = 0.0

    for experience in experience_array:
        # Edge case - start_date is none / null, skip
        if not experience.date_start:
            continue

        start = parse_flexible_date(experience.date_start)
        end   = parse_flexible_date(experience.date_end)

        # Calculate the time difference between the start and end time 
        time_diff = get_time_difference(start, end)

        total_yoe += time_diff

    # YoE cannot be less than 0
    total_yoe = max(0, total_yoe)

    return total_yoe

@udf(FloatType())
def get_title_similarity_score(jd_job_title, experience_array) -> float:
    """
    Get the sim score match between the jd title and the experience array
    """
    # If JD job title is None, just return 0.0
    if not jd_job_title:
        print(0.0)
        return 0.0

    jd_job_title_embedding = embedding_model.encode([jd_job_title],
                                                    convert_to_tensor=True,
                                                    normalize_embeddings=True)
    
    # Need to handle None values
    exp_job_titles        = [exp.role for exp in experience_array if exp.role]

    #If no exp_job titles, nothing to compare to, so just 0
    if len(exp_job_titles) == 0:
        print(0.0)
        return 0.0

    experience_embeddings = embedding_model.encode(exp_job_titles,
                                                    convert_to_tensor=True,
                                                    normalize_embeddings=True)

    similarity_matrix = util.cos_sim(jd_job_title_embedding, experience_embeddings)
    average_score     = torch.mean(similarity_matrix).item()

    print(average_score)

    return average_score

def data_processing_silver_table(datamart_dir : str, selected_date : str, spark : SparkSession) -> None:
    """
    Merge the parquets together, get the dataframe for further processing
    """
    jd_full_dir     = os.path.join(datamart_dir, "bronze", f"jd_{selected_date}.parquet")
    resume_full_dir = os.path.join(datamart_dir, "bronze", f"resume_{selected_date}.parquet")
    labels_full_dir = os.path.join(datamart_dir, "bronze", f"labels_{selected_date}.parquet")

    jd_df     = spark.read.parquet(jd_full_dir)
    resume_df = spark.read.parquet(resume_full_dir)
    labels_df = spark.read.parquet(labels_full_dir)

    # We do the individual transforms to dfs first
    ## Resume Transforms
    resume_df = resume_df.withColumn("YoE", get_resume_yoe(resume_df['experience']))
    resume_df = resume_df.withColumnRenamed("certifications", "resume_certifications")
    resume_df = resume_df.withColumnRenamed("id", "resume_id")
    resume_df = resume_df.withColumnRenamed("snapshot_date", "resume_snapshot")

    ## JD Transforms
    jd_df = jd_df.withColumnRenamed("certifications", "jd_certifications")
    jd_df = jd_df.withColumnRenamed("id", "job_id")
    jd_df = jd_df.withColumnRenamed("snapshot_date", "job_snapshot")

    # Combine the parquets together
    labels_jd = labels_df.join(jd_df, on="job_id", how="inner")
    labels_jd_resume = labels_jd.join(resume_df, on="resume_id", how="inner")

    """
    DO UDFS HERE
    """
    # Get the experience similarity score 
    # labels_jd_resume = labels_jd_resume.withColumn("exp_sim", get_title_similarity_score(labels_jd_resume['role_title'], labels_jd_resume['experience']))

    # check
    print(f"Silver Table Snapshot : {selected_date} No. Rows : {labels_jd_resume.count()}")

    # Save the parquet 
    filename    = "labels_" + selected_date + ".parquet"
    output_path = os.path.join(datamart_dir, "silver", filename)
    labels_jd_resume.write.mode("overwrite").parquet(output_path)