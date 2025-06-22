import os
from dotenv import load_dotenv
import argparse

from pyspark.sql import SparkSession

from utils.mongodb_utils import get_pyspark_session
from utils.date_utils import *
from utils.data_processing_bronze_table import process_bronze_table
from utils.data_processing_silver_table import *



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data.")
    parser.add_argument('--start', type=int, required=True, help='Start index')
    parser.add_argument('--end', type=int, required=True, help='End index')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for bronze table processing')
    args = parser.parse_args('--type', type=int, default=1, help='Inference or training')

    load_dotenv()
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    spark = get_pyspark_session()


    # Get the range of dates
    date_range = get_snapshot_dates(datetime(2022, 6, 1), datetime(2022, 6, 1))
    process_bronze_table(spark, 0, 1000, 10, "inference")

    # process_bronze_table(spark, args.start, args.end, args.batch_size, "training")

    # For each range, read the silver table and parse
    for cur_date in date_range:
        snapshot_date = f"{cur_date.year}-{cur_date.month:02d}"
        print("Processing silver {}".format(snapshot_date))
        resume_df = data_processing_silver_resume(cur_date, spark, "inference")
        jd_df = data_processing_silver_jd(cur_date, spark, "inference")
        data_processing_silver_labels(cur_date, spark)
        data_processing_silver_combined(cur_date, spark, "inference", resume_df, jd_df)
    