"""
File includes functions to extract educational features from the combined silver dataset & engineers gold level features

Input features from silver dataset:
- resume features:
    - edu_highest_level: highest level of education achieved (e.g., Bachelor's, Master's)
    - edu_field: field of study (e.g., Computer Science, Engineering)
    - edu_gpa: GPA (e.g., 3.5)
    - edu_institution: name of the educational institution (e.g., Stanford University)
    - cert_categories: list of certification categories (e.g., AWS, Azure)
- jd features:
    - edu_required_level: required level of education (e.g., Bachelor's, Master's)
    - edu_required_field: required field of study (e.g., Computer Science, Engineering)
    - required_cert_categories: list of required certification categories (e.g., AWS, Azure)

Output features for gold dataset:
- edu_level_match: boolean indicating if the resume's highest education level matches the JD's required level
- edu_level_score: float between 0 and 1 {1: exact match, 0.75: overqualified, 0: underqualified} indicating the match score based on education level
- edu_field_match: boolean indicating the match between the resume's field of study and the JD's required field
- edu_gpa: float between 0 and 1 indicating performance based on GPA, larger is better
- edu_institution: float between 0 and 1 indicating the prestige of the institution, larger is better
- cert_match: boolean indicating if the resume's certifications match the JD's required certifications
"""

"""
This script contains the logic to generate all education-related gold features.
It is designed to be called from the main data processing pipeline.
"""
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType, FloatType, BooleanType, StructType, StructField
from rapidfuzz import fuzz
import re

# ==============================================================================
#  INTERNAL HELPER FUNCTIONS
# ==============================================================================

def _clean_institution_name(name: str) -> str:
    """Cleans a university name to a standardized format for matching."""
    if name is None:
        return None
    name = name.lower()
    name = re.sub(r'[^\w\s]', '', name)
    common_words = ['the', 'of', 'and', 'university', 'college', 'institute']
    query = name.split()
    result_words = [word for word in query if word.lower() not in common_words]
    name = ' '.join(result_words)
    name = name.strip()
    return name

def _standardize_gpa(df: DataFrame) -> DataFrame:
    """Standardizes the 'edu_gpa' column in a DataFrame to a 0-4 scale."""
    return df.withColumn(
        "edu_gpa",
        F.when(F.col("edu_gpa").isNull(), F.lit(None))
         .when(F.col("edu_gpa") <= 4, F.col("edu_gpa"))
         .when((F.col("edu_gpa") > 4) & (F.col("edu_gpa") <= 10), (F.col("edu_gpa") / 10.0) * 4.0)
         .when((F.col("edu_gpa") > 10) & (F.col("edu_gpa") <= 100), (F.col("edu_gpa") / 100.0) * 4.0)
         .otherwise(F.lit(None))
         .cast(FloatType())
    )

def _create_institution_tier(spark: SparkSession, df: DataFrame) -> DataFrame:
    """Adds an 'institution_tier' column by matching against a clean Parquet reference file."""
    rankings_parquet_path = "../../datamart/references/qs_rankings"
    
    # 1. Load and Prepare Rankings Data
    try:
        rankings_df = spark.read.parquet(rankings_parquet_path)
    except Exception as e:
        print(f"  WARNING: Could not read rankings data from {rankings_parquet_path}. Skipping institution tiering. Error: {e}")
        return df.withColumn("institution_tier", F.lit("Tier 3")) # Default to Tier 3 if rankings are missing

    clean_name_udf = F.udf(_clean_institution_name, StringType())
    rankings_df = rankings_df.select("Institution", "Rank").withColumn("cleaned_qs_name", clean_name_udf(F.col("Institution")))

    # 2. Prepare Resume Data
    resume_institutions_df = df.select("edu_institution").filter(F.col("edu_institution").isNotNull()).distinct()
    resume_institutions_df = resume_institutions_df.withColumn("cleaned_resume_name", clean_name_udf(F.col("edu_institution")))

    # 3. Fuzzy Match
    def get_best_match(resume_name, rankings_broadcast):
        best_score, best_rank = 0, None
        for row in rankings_broadcast.value:
            score = fuzz.token_sort_ratio(resume_name, row['cleaned_qs_name'])
            if score > best_score:
                best_score, best_rank = score, row['Rank']
        return (best_rank, best_score) if best_score > 85 else (None, 0)

    rankings_list_broadcast = spark.sparkContext.broadcast(rankings_df.collect())
    match_schema = StructType([
        StructField("matched_rank", IntegerType(), True),
        StructField("match_score", IntegerType(), True)
    ])
    get_best_match_udf = F.udf(lambda name: get_best_match(name, rankings_list_broadcast), match_schema)
    
    matched_institutions_df = resume_institutions_df.withColumn(
        "match_result", get_best_match_udf(F.col("cleaned_resume_name"))
    ).select(
        "edu_institution",
        F.col("match_result.matched_rank").alias("matched_rank")
    )
    
    # 4. Join and Create Tier
    df_with_rank = df.join(matched_institutions_df, "edu_institution", "left")
    df_with_tier = df_with_rank.withColumn(
        "institution_tier",
        F.when(F.col("matched_rank").isNotNull() & (F.col("matched_rank") <= 100), "Tier 1")
         .when(F.col("matched_rank").isNotNull() & (F.col("matched_rank") <= 500), "Tier 2")
         .otherwise("Tier 3")
    )
    
    return df_with_tier.drop("matched_rank")

def map_edu_level(req: str, given: str, scale: list):
    if req is None:
        if given is None:
            return (None, None)
        else:
            # No requirement, but education provided.
            return (True, 0.75)
    # From here, req is not None.
    if given is None:
        return (False, 0.0) # Requirement, but no education provided.
    req_n_list = [row.group_scale for row in scale if row.group_name == req]
    given_n_list = [row.group_scale for row in scale if row.group_name == given]
    # If required or given education level is not in our reference scale, we can't compare.
    if not req_n_list or not given_n_list:
        return (False, 0.0)
    req_n = req_n_list[0]
    given_n = given_n_list[0]
    if given_n == req_n:
        return (True, 1.0)
    elif given_n > req_n:
        return (True, 0.75)
    else: # given_n < req_n
        return (False, 0.0)

def map_edu_field(req: str, given: str) -> bool:
    if req is None or req == "Others":
        return True
    elif req == given:
        return True
    else:
        return False

def map_certification(req: list, given: list) -> bool:
    if req is None or not req or len(req) == 0:
        return True
    # from here, req is not None and has at least one element.
    if given is None or not given or len(given) == 0:
        return False
    # from here, req and given are not None and have at least one element.
    # if any given certification matches any required certification, return True.
    for r in req:
        if r in given:
            return True
    return False

# ==============================================================================
#  MAIN PUBLIC FUNCTION TO BE IMPORTED
# ==============================================================================
def extract_education_features(df: DataFrame) -> DataFrame:
    """
    Main entry point function to run all education-related feature extractions.

    Args:
        df (DataFrame): The input DataFrame from the previous processing step.

    Returns:
        DataFrame: DataFrame with new education features added.
    """
    print("  Extracting education features...")
    
    # Get the active SparkSession
    spark = SparkSession.builder.getOrCreate()

    # Match education level
    print("  Matching education levels...")

    # Broadcast the education level synonyms scale
    scale = spark.sparkContext.broadcast(spark.read.parquet("datamart/references/education_level_synonyms.parquet").collect())
    # Define the schema for the temporary UDF output
    EDU_TMP_SCHEMA = StructType([
        StructField("edu_match", BooleanType(), True),
        StructField("edu_score", FloatType(), True),
    ])
    # map education levels
    df = (
        df
        .withColumn("tmp_edu_match", F.udf(lambda req, giv: map_edu_level(req, giv, scale.value), EDU_TMP_SCHEMA)(F.col("required_edu_level"), F.col("edu_highest_level")))
        .withColumn("edu_level_match", F.col("tmp_edu_match.edu_match"))
        .withColumn("edu_level_score", F.col("tmp_edu_match.edu_score"))
        .drop("tmp_edu_match", "required_edu_level", "edu_highest_level")
    )
    print("Distinct edu_level_match values:")
    for match in df.select('edu_level_match').distinct().collect():
        c = df.filter(df.edu_level_match == match.edu_level_match).count()
        print(f"{match.edu_level_match} : {c}")
    print("Distinct edu_level_score values:")
    for score in df.select('edu_level_score').distinct().collect():
        c = df.filter(df.edu_level_score == score.edu_level_score).count()
        print(f"{score.edu_level_score} : {c}")

    # Match education field
    print("  Matching education fields...")

    df = (
        df
        .withColumn("edu_field_match", F.udf(map_edu_field, BooleanType())(F.col("required_edu_field"), F.col("edu_field")))
        .drop("required_edu_field", "edu_field")
    )
    print("Distinct edu_field_match values:")
    for match in df.select('edu_field_match').distinct().collect():
        c = df.filter(df.edu_field_match == match.edu_field_match).count()
        print(f"{match.edu_field_match} : {c}")
    
    # Match certifications
    print("  Matching certifications...")
    df = (
        df
        .withColumn("cert_match", F.udf(map_certification, BooleanType())(F.col("required_cert_categories"), F.col("cert_categories")))
        .drop("required_cert_categories", "cert_categories")
    )
    print("Distinct cert_match values:")
    for match in df.select('cert_match').distinct().collect():
        c = df.filter(df.cert_match == match.cert_match).count()
        print(f"{match.cert_match} : {c}")

    # GPA standardization
    print("  Standardizing GPA...")
    df_with_gpa = _standardize_gpa(df)

    # Institution tiering
    print("  Creating institution tiers...")
    df_with_all_edu_features = _create_institution_tier(spark, df_with_gpa)
    
    print("  Education features extracted.")
    return df_with_all_edu_features