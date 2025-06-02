import pandas as pd
import os
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
from dotenv import load_dotenv
import json
from tqdm import tqdm
import string
import re

from .resume_schema import Resume
from .jd_schema import JD
from pydantic import BaseModel, Field
from typing import List, Optional, get_origin, get_args, Union

from langchain.output_parsers import PydanticOutputParser
from mistralai import Mistral

from pyspark.sql.types import ArrayType, StringType, IntegerType, FloatType, BooleanType, TimestampType, StructField, StructType

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
def process_bronze_table(spark, partition_start, partition_end, batch_size):
    print("============ PROCESS BRONZE TABLE =============")
    ###############
    # Retrieve data from source
    ###############
    df = retrieve_data_from_source()
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

    df_parted = df.iloc[partition_start:partition_end]

    batch_idx = 0
    
    for idx, row in tqdm(df_parted.iterrows(), total=len(df_parted), desc=f"Processing indexes {partition_start}-{partition_end - 1}"):
        resume_text = row['resume_text']
        jd_text = row['job_description_text']
        try:
            # Process resume
            parsed_resume = parse_with_llm(resume_text, resume_parser, "Resume", llm)
            parsed_resume_dict = parsed_resume.model_dump(mode="json")
            parsed_resume_dict['snapshot_date'] = row['snapshot_date_str']
            parsed_resume_dict['id'] = row['resume_id']
            parsed_resumes.append(parsed_resume_dict)

            # Process JD
            parsed_jd = parse_with_llm(jd_text, jd_parser, "Job Description", llm)
            parsed_jd_dict = parsed_jd.model_dump(mode="json")
            parsed_jd_dict['snapshot_date'] = row['snapshot_date_str']
            parsed_jd_dict['id'] = row['job_id']
            parsed_jds.append(parsed_jd_dict)
            
            batch_idx += 1

            if batch_idx == batch_size:
                resume_df = spark.createDataFrame(parsed_resumes, schema=pydantic_to_spark_schema(Resume))
                jd_df = spark.createDataFrame(parsed_jds, schema=pydantic_to_spark_schema(JD))

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
                
                parsed_resumes.clear()
                parsed_jds.clear()
                batch_idx = 0
        
        except Exception as e:
            print(f"Error parsing row {idx}: {e}")
    print(f"============ END PROCESS BRONZE TABLE =============")
