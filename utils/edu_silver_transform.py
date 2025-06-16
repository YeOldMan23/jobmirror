"""
This is a stand alone util file to test the education extraction for the resume 

this can be run from test_silver_table.py by uncommenting the line below within the same file

resume_edu_to_silver(datamart_dir, snapshot_date, spark)            # testing for education


"""








import os, re
from datetime import datetime
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from utils.edu_utils import level_from_text, major_from_text, gpa_from_text

_EDU_SCHEMA = StructType([
    StructField("highest_level_education", StringType(), True),
    StructField("major",                   StringType(), True),
    StructField("gpa",                     FloatType(),  True),
    StructField("institution",             StringType(), True),
])

def _parse_edu(arr):
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

_edu_udf = udf(_parse_edu, _EDU_SCHEMA)

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
