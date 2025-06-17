"""
This is a stand alone util file to test the education extraction for the resume 

this can be run from test_silver_table.py by uncommenting the line below within the same file

resume_edu_to_silver(datamart_dir, snapshot_date, spark)            # testing for education


"""








import os, re
from datetime import datetime
import pyspark
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType
from utils.edu_utils import level_from_text, major_from_text, gpa_from_text, determine_edu_mapping

_EDU_SCHEMA = StructType([
    StructField("highest_level_education", StringType(), True),
    StructField("major",                   StringType(), True),
    StructField("gpa",                     FloatType(),  True),
    StructField("institution",             StringType(), True),
])

def _parse_cv_edu(arr):
    if not arr:
        return (None, None, None, None)
    ed0 = arr[0]
    txt = f"{ed0.degree or ''} {ed0.description or ''}"
    lvl, _ = level_from_text(txt)
    maj, _ = major_from_text(txt)
    gpa = ed0.grade or gpa_from_text(txt)
    return (
        lvl,
        maj,
        float(gpa) if gpa is not None else None,
        ed0.institution,
    )

_edu_udf = udf(_parse_cv_edu, _EDU_SCHEMA)

def resume_edu_to_silver(datamart_dir: str, snap: str, spark: SparkSession) -> None:
    in_path  = os.path.join(datamart_dir, "bronze", f"resume_{snap}.parquet")
    out_path = os.path.join(datamart_dir, "silver", f"resume_silver_{snap}.parquet")

    if not os.path.exists(in_path):
        print(f"⤬ resume parquet not found for {snap}")
        return

    df = spark.read.parquet(in_path)

    df = (
        df
        .withColumn("edu_tmp", _edu_udf("education"))
        .withColumn("highest_level_education", col("edu_tmp.highest_level_education"))
        .withColumn("major",                   col("edu_tmp.major"))
        .withColumn("gpa",                     col("edu_tmp.gpa"))
        .withColumn("institution",             col("edu_tmp.institution"))
        .drop("edu_tmp")
    )

    df.write.mode("overwrite").parquet(out_path)
    print(f"✓ wrote {out_path}  rows={df.count()}")


def jd_edu_to_silver(spark: SparkSession, datamart_dir: str, snap: str) -> None:
    """
    Transform the job description education data to silver table format.
    Args:
        df (DataFrame): The bronze level DataFrame containing job description data with education information.
    Returns:
        df (DataFrame): The transformed DataFrame with education information extracted and structured.
    """

    in_path  = os.path.join(datamart_dir, "bronze", f"jd_{snap}.parquet")
    out_path = os.path.join(datamart_dir, "silver", f"jd_silver_{snap}.parquet")

    if not os.path.exists(in_path):
        print(f"⤬ jd parquet not found for {snap}")
        return

    df = spark.read.parquet(in_path)

    edu_levels = spark.sparkContext.broadcast(spark.read.parquet('data/education_level_synonyms.parquet').collect())
    edu_fields = spark.sparkContext.broadcast(spark.read.parquet('data/education_field_synonyms.parquet').collect())
    cert_categories = spark.sparkContext.broadcast(spark.read.parquet('data/certification_categories.parquet').collect())

    df = (
        df
        .withColumn("required_edu_level", udf(lambda x: determine_edu_mapping(x, edu_levels.value), StringType())("required_education"))
        .withColumn("required_edu_field", udf(lambda x: determine_edu_mapping(x, edu_fields.value), StringType())("required_education"))
        .withColumn("required_cert_field", udf(lambda x: determine_edu_mapping(x, cert_categories.value), StringType())("jd_certifications"))
        .withColumn("no_of_certs", udf(lambda x: len(x) if isinstance(x, list) else 0, IntegerType())("jd_certifications"))
        .drop("required_education")
        .drop("jd_certifications")
    )

    df.write.mode("overwrite").parquet(out_path)
    print(f"✓ wrote {out_path}  rows={df.count()}")