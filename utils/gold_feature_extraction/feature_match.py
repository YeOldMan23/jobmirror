from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, expr, size, array_max, lit, transform, struct

def match_employment_type(df):
    df = df.withColumn(
        "employment_type_match",
        (col("employment_type_preference") == col("employment_type")).cast("integer")
    )

    return df

def match_work_authorization(df):
    df.withColumn(
        "work_authorization_match",
        (col("work_authorization") == col("required_work_authorization")).cast("integer")
    )

    return df

def match_location_preference(df):
    df = df.withColumn(
        "location_preference_match",
        (col("location_preference") == col("job_location")).cast("integer")
    )

    return df
