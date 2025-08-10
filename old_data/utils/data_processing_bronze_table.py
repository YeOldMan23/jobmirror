import pandas as pd
import os
import sys
# Ensure /opt/airflow/utils is in sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
from dotenv import load_dotenv
import json
from tqdm import tqdm
import string
import re
import time
import argparse

from resume_schema import Resume
from jd_schema import JD
from pydantic import BaseModel, Field
from typing import List, Optional, get_origin, get_args, Union

from langchain.output_parsers import PydanticOutputParser
from mistralai import Mistral

from pyspark.sql.types import ArrayType, StringType, IntegerType, FloatType, BooleanType, TimestampType, StructField, StructType
from mongodb_utils import get_collection, exists_in_collection
from mongodb_utils import get_pyspark_session
# from utils.s3_utils import upload_to_s3
from date_utils import *

###############
# SOURCE
###############
def retrieve_data_from_source():
    """
    Reads data as Pandas dataframe from source
    """
    def generate_random_snapshot_dates(df):
        rng = np.random.default_rng(seed=42)
        # Define start and end date
        start_date = pd.to_datetime('2021-06-01')
        end_date = pd.to_datetime('2021-12-1')
        # Generate random timestamps between start_date and end_date
        random_dates = pd.to_datetime(
            rng.uniform(start_date.value, end_date.value, size=len(df))
        )
        # Ensure it's treated as a pandas Series and convert to date
        df['snapshot_date'] = pd.Series(random_dates).dt.date  # This will convert to date format
        df['snapshot_date_str'] = df['snapshot_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        return df

    def generate_random_id(prefix, seed, length=8, use_digits=True, use_letters=True):
        rng = np.random.default_rng(seed=seed) 

        characters = ''
        
        if use_digits:
            characters += string.digits
        if use_letters:
            characters += string.ascii_letters

        # Ensure we have characters to choose from
        if not characters:
            raise ValueError("At least one of 'use_digits' or 'use_letters' must be True.")
        
        # Use np.random.choice to randomly select characters
        random_id = ''.join(rng.choice(list(characters), size=length))
        return prefix + random_id
    
    # Download from huggingface
    splits = {'train': 'train.csv', 'test': 'test.csv'}
    df = pd.read_csv("hf://datasets/cnamuangtoun/resume-job-description-fit/" + splits["train"])

    # Generate random snapshot dates
    df = generate_random_snapshot_dates(df)

    # Generate random ids
    df['resume_id'] = df.apply(lambda row: generate_random_id('RES_', seed=row.name), axis=1)
    df['job_id'] = df.apply(lambda row: generate_random_id('JD_', seed=row.name), axis=1)
    df['label_id'] = df.apply(lambda row: generate_random_id('LABEL_', seed=row.name), axis=1)
    
    return df

def retrieve_inference_data():
    """
    Reads data as Pandas dataframe from source
    """
    def generate_random_snapshot_dates(df):
        rng = np.random.default_rng(seed=42)
        # Define start and end date
        start_date = pd.to_datetime('2022-06-01')
        end_date = pd.to_datetime('2022-12-1')
        # Generate random timestamps between start_date and end_date
        random_dates = pd.to_datetime(
            rng.uniform(start_date.value, end_date.value, size=len(df))
        )
        # Ensure it's treated as a pandas Series and convert to date
        df['snapshot_date'] = pd.Series(random_dates).dt.date  # This will convert to date format
        df['snapshot_date_str'] = df['snapshot_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        return df

    def generate_random_id(prefix, seed, length=8, use_digits=True, use_letters=True):
        rng = np.random.default_rng(seed=seed) 

        characters = ''
        
        if use_digits:
            characters += string.digits
        if use_letters:
            characters += string.ascii_letters

        # Ensure we have characters to choose from
        if not characters:
            raise ValueError("At least one of 'use_digits' or 'use_letters' must be True.")
        
        # Use np.random.choice to randomly select characters
        random_id = ''.join(rng.choice(list(characters), size=length))
        return prefix + random_id
    
    # Download from huggingface
    splits = {'train': 'train.csv', 'test': 'test.csv'}
    df = pd.read_csv("hf://datasets/cnamuangtoun/resume-job-description-fit/" + splits["test"])

    # Generate random snapshot dates
    df = generate_random_snapshot_dates(df)

    # Generate random ids
    df['resume_id'] = df.apply(lambda row: generate_random_id('RES_', seed=(6241 + row.name)), axis=1)
    df['job_id'] = df.apply(lambda row: generate_random_id('JD_', seed=(6241 + row.name)), axis=1)
    df['label_id'] = df.apply(lambda row: generate_random_id('LABEL_', seed=(6241 + row.name)), axis=1)
    
    return df

###############
# UTILITY FUNCTIONS
###############
def clean_text(text):
    # 1. Remove non-ASCII characters
    text = text.encode('ascii', 'ignore').decode()

    # 2. Keep only alphanumeric, punctuation, and whitespace
    text = re.sub(r'[^a-zA-Z0-9\s!"#$%&\'()*+,\-./:;<=>?@[\\\]^_`{|}~]', '', text)

    return text

def parse_with_llm(text, parser, label, mistral):
    """
    Function to parse LLM into Pydantic schema
    """
    prompt = (
        f"Parse the following text into a structured format according to the provided schema."
        f"If the same role at the same company appears more than once, merge the role descriptions and preserve the earliest start and latest end dates."
        f"{parser.get_format_instructions()}\n\n"
        f"{label}:\n{text}"
    )

    response = mistral.chat.complete(
        model="mistral-medium-latest",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=2048
    )
    raw = response.choices[0].message.content
    return parser.parse(raw)

# create label structure and schema 
def create_label_dicts(df):
    label_dicts = []
    for _, row in df.iterrows():
        label_dicts.append({
            "label_id": row['label_id'],
            "resume_id": row['resume_id'],
            "job_id": row['job_id'],
            "fit": row['label'],
            "snapshot_date": str(row['snapshot_date'])
        })
    return label_dicts

label_schema = StructType([
    StructField("id", StringType(), True),
    StructField("resume_id", StringType(), True),
    StructField("job_id", StringType(), True),
    StructField("fit", StringType(), True), # 
    StructField("snapshot_date", StringType(), True),
])

def python_type_to_spark_type(annotation):
    """
    Convert Python data type to Spark data type
    """
    origin = get_origin(annotation)

    if origin is Union:  # Handle Optional
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        return python_type_to_spark_type(args[0])

    if origin in (list, List):
        element_type = python_type_to_spark_type(get_args(annotation)[0])
        return ArrayType(element_type)

    if isinstance(annotation, type):
        if issubclass(annotation, BaseModel):
            return pydantic_to_spark_schema(annotation)
        if issubclass(annotation, str):
            return StringType()
        if issubclass(annotation, int):
            return IntegerType()
        if issubclass(annotation, float):
            return FloatType()
        if issubclass(annotation, bool):
            return BooleanType()
        if issubclass(annotation, datetime.datetime):
            return StringType()

    return StringType()

def pydantic_to_spark_schema(model: type) -> StructType:
    """
    Convert Pydantic schema to PySpark schema
    """
    fields = []

    for name, field in model.model_fields.items():
        annotation = field.annotation

        spark_type = python_type_to_spark_type(annotation)
        fields.append(StructField(name, spark_type, True))  # assume all nullable
    fields.append(StructField('snapshot_date', StringType(), True))
    fields.append(StructField('id', StringType(), True))

    return StructType(fields)

###############
# MAIN FUNCTION
###############
def process_bronze_table(spark, partition_start, partition_end, batch_size, type):
    print("============ PROCESS BRONZE TABLE =============")

    ###############
    # Retrieve data from source
    ###############
    # if type == "training":
        # df = retrieve_data_from_source()
        
    resume_collection = get_collection("jobmirror_db", "bronze_resumes")
    label_collection = get_collection("jobmirror_db", "bronze_labels")
    jd_collection = get_collection("jobmirror_db", "bronze_job_descriptions")

    # elif type == "inference":
    #     # df = retrieve_inference_data()
    #     resume_collection = get_collection("jobmirror_db", "online_bronze_resumes")
    #     label_collection = get_collection("jobmirror_db", "online_bronze_labels")
    #     jd_collection = get_collection("jobmirror_db", "online_bronze_job_descriptions")

    # Load documents as DataFrames
    resumes = pd.DataFrame(list(resume_collection.find()))
    labels = pd.DataFrame(list(label_collection.find()))
    jds = pd.DataFrame(list(jd_collection.find()))

    # Merge based on IDs
    df = labels.merge(resumes, on="resume_id").merge(jds, on="job_id")

    df['resume_text'] = df['resume_text'].apply(clean_text)
    df['job_description_text'] = df['job_description_text'].apply(clean_text)

    print("Retrieved data from source")

    ###############
    # Define LLM
    ###############
    llm = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))

    ###############
    # Define output parsers
    ###############
    resume_parser = PydanticOutputParser(pydantic_object=Resume)
    jd_parser = PydanticOutputParser(pydantic_object=JD)

    ###############
    # Parse every row in df
    ###############

    parsed_resumes = []
    parsed_jds = []
    parsed_labels = []

    resume_collection = get_collection("jobmirror_db", "bronze_resumes")
    label_collection = get_collection("jobmirror_db", "bronze_labels")
    jd_collection = get_collection("jobmirror_db", "bronze_job_descriptions")

    df_parted = df.iloc[partition_start:partition_end]

    batch_idx = 0

    for idx, row in tqdm(df_parted.iterrows(), total=len(df_parted), desc=f"Processing indexes {partition_start}-{partition_end - 1}"):
        resume_text = row['resume_text']
        jd_text = row['job_description_text']

        try:
            # Process resume
            if exists_in_collection(resume_collection, row['resume_id']):
                print(f"Resume with id {row['resume_id']} already exists. Skipping.")
            else:
                parsed_resume = parse_with_llm(resume_text, resume_parser, "Resume", llm)
                parsed_resume_dict = parsed_resume.model_dump(mode="json")
                parsed_resume_dict['snapshot_date'] = row['snapshot_date_str']
                parsed_resume_dict['id'] = row['resume_id']
                parsed_resumes.append(parsed_resume_dict)

            # Process JD
            if exists_in_collection(jd_collection, row['job_id']):
                print(f"JD with id {row['job_id']} already exists. Skipping.")
            else:
                parsed_jd = parse_with_llm(jd_text, jd_parser, "Job Description", llm)
                parsed_jd_dict = parsed_jd.model_dump(mode="json")
                parsed_jd_dict['snapshot_date'] = row['snapshot_date_str']
                parsed_jd_dict['id'] = row['job_id']
                parsed_jds.append(parsed_jd_dict)

            # Process label
            if exists_in_collection(label_collection, row['label_id']):
                print(f"Label with id {row['label_id']} already exists. Skipping.")
            else:
                parsed_labels.append({
                    "id": row['label_id'],
                    "resume_id": row['resume_id'],
                    "job_id": row['job_id'],
                    "fit": row['label'],  
                    "snapshot_date": str(row['snapshot_date'])
                })

            batch_idx += 1

            if batch_idx == batch_size:
                resume_df = spark.createDataFrame(parsed_resumes, schema=pydantic_to_spark_schema(Resume))
                jd_df = spark.createDataFrame(parsed_jds, schema=pydantic_to_spark_schema(JD))
                label_df = spark.createDataFrame(parsed_labels, schema=label_schema)
                # if type == "training":
                resume_df.write.format("mongodb") \
                    .mode("append") \
                    .option("database", "jobmirror_db") \
                    .option("collection", "bronze_resumes") \
                    .save()

                jd_df.write.format("mongodb") \
                    .mode("append") \
                    .option("database", "jobmirror_db") \
                    .option("collection", "bronze_job_descriptions") \
                    .save()

                label_df.write.format("mongodb") \
                    .mode("append") \
                    .option("database", "jobmirror_db") \
                    .option("collection", "bronze_labels") \
                    .save()

                parsed_resumes.clear()
                parsed_jds.clear()
                parsed_labels.clear()

                # elif type == "inference":
                #     resume_df.write.format("mongodb") \
                #         .mode("append") \
                #         .option("database", "jobmirror_db") \
                #         .option("collection", "online_bronze_resumes") \
                #         .save()

                #     jd_df.write.format("mongodb") \
                #         .mode("append") \
                #         .option("database", "jobmirror_db") \
                #         .option("collection", "online_bronze_job_descriptions") \
                #         .save()

                #     label_df.write.format("mongodb") \
                #         .mode("append") \
                #         .option("database", "jobmirror_db") \
                #         .option("collection", "online_bronze_labels") \
                #         .save()

                    # # Get snapshot_date from the row data
                    # snapshot_date = row['snapshot_date']
                    # inf_list = [("resume", resume_df), ("jd", jd_df), ("labels", label_df)]
                    
                    # for f, df_to_write in inf_list:
                    #     filename = f"{snapshot_date.year}-{snapshot_date.month:02d}.parquet"
                    #     s3_key = f"datamart/online/bronze/{f}/{filename}"
                    #     output_path = os.path.join("datamart", "silver", f, filename)
                        
                    #     df_to_write.write.mode("overwrite").parquet(output_path)
                    #     upload_to_s3(output_path, s3_key)
                    
                    # parsed_resumes.clear()
                    # parsed_jds.clear()
                    # parsed_labels.clear()                    

                batch_idx = 0

            if idx % 100 == 0:
                time.sleep(1.5)
        
        except Exception as e:
            print(f"Error parsing row {idx}: {e}")

    print(f"============ END PROCESS BRONZE TABLE =============")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data.")
    parser.add_argument('--start', type=int, required=True, help='Start index')
    parser.add_argument('--end', type=int, required=True, help='End index')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for bronze table processing')
    parser.add_argument('--skip', type=bool, default=True, help='Skip bronze table processing')
    parser.add_argument('--type', type=str, default='training', help='Inference or training')
    
    args = parser.parse_args()  # Fixed - removed the extra argument definition

    try:
        if not args.skip:
            load_dotenv("/opt/airflow/.env")
            os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
            spark = get_pyspark_session()
            # Get the range of dates
            process_bronze_table(spark, args.start, args.end, args.batch_size, args.type) 
    except Exception as e:
        print("An error occurred:", e)

