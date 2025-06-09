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
embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

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

@udf(FloatType())
def get_resume_yoe(experience_array) -> float:
    """
    Get the YoE from resume experiences
    """
    total_yoe = 0.0

    for experience in experience_array:
        start = datetime.fromisoformat(experience.get('date_start'))
        end   = experience.get('date_end')
        
        # Edge case - job is current date, just set the job experience as end of 2021
        pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$"
        if re.match(pattern, end):
            # it is proper ISO format
            end = datetime.fromisoformat(end)
        else:
            # it is current job, just set end time as last day of 2021
            end = datetime(2021, 12, 31)

        # Calculate the time difference between the start and end time 
        time_diff = get_time_difference(start, end)

        total_yoe += time_diff

    return total_yoe

@udf(FloatType())
def get_title_similarity_score(jd_job_title, experience_array) -> float:
    """
    Get the sim score match between the jd title and the experience array
    """
    jd_job_title_embedding = embedding_model.encode([jd_job_title],
                                                    convert_to_tensor=True,
                                                    normalize_embeddings=True)
    experience_embeddings = []
    for experience in experience_array:
        exp_job_title = experience.get("role")
        experience_embedding = embedding_model.encode([exp_job_title],
                                                      convert_to_tensor=True,
                                                      normalize_embeddings=True)
        experience_embeddings.append(experience_embedding)

    similarity_matrix = util.cos_sim(jd_job_title_embedding, experience_embeddings)
    average_score     = torch.mean(similarity_matrix).item()

    return average_score

def merge_parquets(datamart_dir : str, selected_date : str, spark : SparkSession) -> DataFrame:
    """
    Merge the parquets together, get the dataframe for further processing
    """
    jd_full_dir     = os.path.join(datamart_dir, f"jd_{selected_date}.parquet")
    resume_full_dir = os.path.join(datamart_dir, f"resume_{selected_date}.parquet")
    labels_full_dir = os.path.join(datamart_dir, f"labels_{selected_date}.parquet")

    jd_df     = spark.read.parquet(jd_full_dir)
    resume_df = spark.read.parquet(resume_full_dir)
    labels_df = spark.read.parquet(labels_full_dir)

    # We do the individual transforms to dfs first
    ## Resume Transforms
    resume_df = resume_df.withColumn("YoE", get_resume_yoe(resume_df['experience']))

    # Combine the parquets together
    labels_jd = labels_df.join(jd_df, labels_df["job_id"] == jd_df["id"], how="inner")
    labels_jd_resume = labels_jd.join(resume_df, labels_jd["resume_id"] == resume_df["id"], how="inner")

    return labels_jd_resume