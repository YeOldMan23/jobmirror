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
from pyspark.sql.functions import udf, col
from pyspark.sql.types import FloatType, StructType, StructField, StringType, ArrayType, BooleanType

# Using another language model to read the values
from sentence_transformers import SentenceTransformer, util

import os


# education helpers
from utils.edu_utils import (
    level_from_text,
    major_from_text,
    gpa_from_text,
)

EDU_OUT_SCHEMA = StructType([
    StructField("highest_level_education", StringType(), True),
    StructField("major",                   StringType(), True),
    StructField("gpa",                     FloatType(),  True),
    StructField("institution",             StringType(), True),
])
def _parse_education(edu_arr):
    if not edu_arr:
        return (None, None, None, None)
    ed0 = edu_arr[0]
    text = f"{ed0.degree or ''} {ed0.description or ''}"
    level, _ = level_from_text(text)
    major, _ = major_from_text(text)
    gpa_val = ed0.grade or gpa_from_text(text)
    return (
        level,
        major,
        float(gpa_val) if gpa_val is not None else None,
        ed0.institution,
    )
parse_education_udf = udf(_parse_education, EDU_OUT_SCHEMA)





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

@udf(FloatType())
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
    resume_df = (
            resume_df
                .withColumn("YoE_list", get_resume_yoe("experience"))
                .withColumnRenamed("certifications", "resume_certifications")
                .withColumnRenamed("id", "resume_id")
                .withColumnRenamed("snapshot_date", "resume_snapshot")
                .withColumn("edu_extracted", parse_education_udf("education"))
                .withColumn("highest_level_education", col("edu_extracted.highest_level_education"))
                .withColumn("major",                   col("edu_extracted.major"))
                .withColumn("gpa",                     col("edu_extracted.gpa"))
                .withColumn("institution",             col("edu_extracted.institution"))
                .drop("edu_extracted")
        )
  
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
    labels_jd_resume = labels_jd_resume.withColumn("exp_sim_list", get_title_similarity_score(labels_jd_resume['role_title'], labels_jd_resume['experience']))

    # check
    print(f"Silver Table Snapshot : {selected_date} No. Rows : {labels_jd_resume.count()}")

    # Save the parquet 
    filename    = "labels_" + selected_date + ".parquet"
    output_path = os.path.join(datamart_dir, "silver", filename)
    labels_jd_resume.write.mode("overwrite").parquet(output_path)