"""
Read the parquet files and extract new information out of them
"""

# Get Spark Session
import pyspark
from pyspark.sql.dataframe import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import FloatType, StructType, StructField, StringType, ArrayType, BooleanType

# Using another language model to read the values
from sentence_transformers import SentenceTransformer, util

import os

import torch
from datetime import datetime
import re

"""
Global Variables
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device Chosen : {}".format(device))
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

@udf(ArrayType(FloatType()))
def get_resume_yoe(experience_array) -> list:
    """
    Get the YoE from resume experiences
    """
    experience_list = []

    for experience in experience_array:
        # Edge case - start_date is none / null, skip
        if not experience.date_start:
            continue

        start = parse_flexible_date(experience.date_start)
        end   = parse_flexible_date(experience.date_end)

        # Calculate the time difference between the start and end time 
        time_diff = get_time_difference(start, end)

        experience_list.append(time_diff)

    return experience_list

@udf(ArrayType(FloatType()))
def get_title_similarity_score(jd_job_title, experience_array) -> list:
    """
    Get the sim score match between the jd title and the experience array
    """
    # If JD job title is None, just return 0.0
    if not jd_job_title:
        # print("No JD title")
        return [-1]

    jd_job_title_embedding = embedding_model.encode([jd_job_title],
                                                    convert_to_tensor=True,
                                                    normalize_embeddings=True)
    
    # Need to handle None values
    exp_job_titles        = [exp.role for exp in experience_array if exp.role]

    #If no exp_job titles, nothing to compare to, so just 0
    if len(exp_job_titles) == 0:
        # print("No experience")
        return []

    experience_embeddings = embedding_model.encode(exp_job_titles,
                                                    convert_to_tensor=True,
                                                    normalize_embeddings=True)

    similarity_matrix = util.cos_sim(jd_job_title_embedding, experience_embeddings).flatten().tolist()

    # print(similarity_matrix)

    return similarity_matrix