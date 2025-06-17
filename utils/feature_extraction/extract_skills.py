"""
Extract and standardize hard skills and soft skills
"""
import os
import glob

from pyspark.sql.functions import expr, udf, explode, collect_list, array
from pyspark.sql import functions as F

from ..gdrive_utils import connect_to_gdrive, sync_gdrive_db_to_local

from pyspark.sql.types import StructType, StructField, ArrayType, StringType, FloatType
from rapidfuzz import process, fuzz
import spacy

def read_hard_skills_list(spark):
    # sync_gdrive_db_to_local()
    
    df_hard_skills_keywords = spark.read.option("header", "true").parquet("datamart/silver/hardskill/Technology_Skills.parquet")
    return df_hard_skills_keywords

def hard_skills_fuzzy_match_udf_factory(mapping_dict):
    def match_array(arr):
        keyword_matches = []
        example_matches = []
        scores = []

        reference_values = list(mapping_dict.keys())
        reference_categories = list(mapping_dict.values())
        
        if arr:
            for x in arr:
                if x:
                    match = process.extractOne(x, reference_values, scorer=fuzz.token_set_ratio)
                    score = float(match[1])
                    if match and score > 80:
                        example_matches.append(match[0])
                        keyword_matches.append(mapping_dict[match[0]])
                        scores.append(score)  # similarity score as float
                    else:
                        match = process.extractOne(x, reference_categories, scorer=fuzz.token_set_ratio)
                        score = float(match[1])
                        if match and score > 80:
                            example_matches.append(None)
                            keyword_matches.append(match[0])
                            scores.append(score)
                        else:
                            example_matches.append(None)
                            keyword_matches.append(None)
                            scores.append(None)
                else:
                    keyword_matches.append(None)
                    example_matches.append(None)
                    scores.append(None)
            return keyword_matches, example_matches, scores
        return None
    
    schema = StructType([
        StructField("keyword_matches", ArrayType(StringType()), nullable=True),
        StructField("example_matches", ArrayType(StringType()), nullable=True),
        StructField("scores", ArrayType(FloatType()), nullable=True)
    ])
    
    return udf(match_array, schema)

def create_hard_skills_column(df_spark, spark, og_column="hard_skills"):
    df_hard_skills_keywords = read_hard_skills_list(spark)
    
    # Convert list of standardized hard skills into keywords
    mapping_dict = dict(df_hard_skills_keywords.rdd.map(lambda row: (row['example'], row['keyword'])).collect())

    # Create udf for fuzzy matching
    fuzzy_match_udf = hard_skills_fuzzy_match_udf_factory(mapping_dict)

    # Rename og column to hard_skills
    df_spark_cleaned = df_spark.withColumnRenamed(og_column, "hard_skills")

    # Convert hard skills column to lowercase
    df_spark_cleaned = df_spark_cleaned.withColumn("hard_skills", expr("transform(hard_skills, x -> lower(x))"))

    # Do the fuzzy matching
    df_spark_cleaned = df_spark_cleaned.withColumn("hard_skills_fuzzy", fuzzy_match_udf("hard_skills"))

    # Split out into individual lists
    df_spark_cleaned = df_spark_cleaned \
        .withColumn("hard_skills_general", df_spark_cleaned["hard_skills_fuzzy"]["keyword_matches"]) \
        .withColumn("hard_skills_specific", df_spark_cleaned["hard_skills_fuzzy"]["example_matches"]) \
        .drop("hard_skills_fuzzy", "hard_skills")
    
    # Remove nulls
    df_spark_cleaned = df_spark_cleaned \
        .withColumn("hard_skills_general", F.expr("filter(hard_skills_general, x -> x is not null)")) \
        .withColumn("hard_skills_specific", F.expr("filter(hard_skills_specific, x -> x is not null)"))
    
    # Remove duplicates
    df_spark_cleaned = df_spark_cleaned \
        .withColumn("hard_skills_general", F.array_distinct("hard_skills_general")) \
        .withColumn("hard_skills_specific", F.array_distinct("hard_skills_specific"))
    
    return df_spark_cleaned

def lemmatize_soft_skills(rows):
    nlp = spacy.load("en_core_web_sm")

    GENERIC_NOUNS = {
        "skill", "skills", "ability", "abilities", "knowledge",
        "proficiency", "understanding", "capability", "competency"
    }

    def extract_head_noun(text):
        doc = nlp(text)
        for chunk in doc.noun_chunks:
            for token in chunk:
                if token.pos_ == "NOUN" and token.lemma_ not in GENERIC_NOUNS:
                    return token.lemma_
        return text

    for row in rows:
        row_dict = row.asDict()
        soft_skills = row_dict.get("soft_skills", [])

        if isinstance(soft_skills, str):
            soft_skills = [soft_skills]
        elif soft_skills is None:
            soft_skills = []

        row_dict["soft_skills_lemmatized"] = [extract_head_noun(s) for s in soft_skills]

        yield tuple(row_dict[col] for col in row_dict.keys())

def create_soft_skills_column(df_spark, spark, og_column="soft_skills"):
    # Check if we need to download the spacy package
    if not spacy.util.is_package("en_core_web_sm"):
        spacy.cli.download("en_core_web_sm")

    # Rename column to soft skills
    df_spark_cleaned = df_spark.withColumnRenamed(og_column, "soft_skills")

    # Convert soft skills column to lowercase
    df_spark_cleaned = df_spark_cleaned.withColumn("soft_skills", expr("transform(soft_skills, x -> lower(x))"))

    # Lemmatize skills
    if not df_spark_cleaned.rdd.isEmpty():
        schema = StructType([*df_spark_cleaned.schema.fields,
                              StructField("soft_skills_lemmatized", ArrayType(StringType()), True)])

        df_spark_cleaned = spark.createDataFrame(df_spark_cleaned.rdd.mapPartitions(lemmatize_soft_skills), schema) \
                            .drop("soft_skills")
        
        df_spark_cleaned = df_spark_cleaned.withColumnRenamed("soft_skills_lemmatized", "soft_skills")

    return df_spark_cleaned



