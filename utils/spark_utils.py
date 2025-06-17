import os
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType

def create_spark_session(app_name: str = "JobMirrorApp") -> SparkSession:
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.mongodb.read.connection.uri", os.environ.get("MONGO_DB_URL")) \
        .config("spark.mongodb.write.connection.uri", os.environ.get("MONGO_DB_URL")) \
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.2.0") \
        .getOrCreate()
    return spark

def create_dataframe(spark: SparkSession, data: list, schema: StructType) -> DataFrame:
    return spark.createDataFrame(data, schema)

def write_to_mongodb(df: DataFrame, database: str, collection: str, mode: str = "overwrite"):
    df.write.format("mongodb") \
        .mode(mode) \
        .option("database", database) \
        .option("collection", collection) \
        .save()


def pyspark_df_info(df: DataFrame):
    """
    Prints information about a PySpark DataFrame, similar to pandas.DataFrame.info().
    Includes total number of rows, and for each column: its name, 
    the number of non-null entries, and its data type.

    Args:
        df: The PySpark DataFrame to get information about.
    """

    if not isinstance(df, DataFrame):
        print("Error: Input is not a PySpark DataFrame.")
        return

    total_rows = df.count()
    print(f"\nTotal entries: {total_rows}")
    
    num_columns = len(df.columns)
    print(f"Data columns (total {num_columns} columns):")
    
    header = f"{'#':<3} {'Column':<25} {'Non-Null Count':<18} {'Dtype':<15}"
    print(header)
    print("--- " + "-"*25 + " " + "-"*18 + " " + "-"*15)

    # get non-null counts for all columns
    agg_exprs = [F.count(c).alias(c) for c in df.columns]
    non_null_counts_row = df.agg(*agg_exprs).collect()[0]

    for i, (col_name, col_type) in enumerate(df.dtypes):
        non_null_count = non_null_counts_row[col_name]
        print(f"{i:<3} {col_name:<25} {non_null_count:<18} {col_type:<15}")
    print("\n")