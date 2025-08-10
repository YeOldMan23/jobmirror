import os
from dotenv import load_dotenv
import argparse

from pyspark.sql import SparkSession

from utils.mongodb_utils import get_pyspark_session
from utils.date_utils import *
from utils.src.data_processing_bronze_table import process_bronze_table
# from utils.src.data_processing_silver_table import *
from utils.mongodb_utils import *

if __name__ == "__main__":
    
    # parser = argparse.ArgumentParser(description="Process data.")
    # parser.add_argument('--start', type=int, required=True, help='Start index')
    # parser.add_argument('--end', type=int, required=True, help='End index')
    # parser.add_argument('--batch_size', type=int, default=1, help='Batch size for bronze table processing')
    # args = parser.parse_args()

    # load_dotenv()
    # os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

    spark = get_pyspark_session()

    # # Get the range of dates
    # date_range = get_snapshot_dates(datetime(2022, 6, 1), datetime(2022, 12, 1))

    # # process_bronze_table(spark, 101, 200, 10, args.type)
    # process_bronze_table(spark, 101, 200, 10, "inference")
    jd = read_bronze_table_as_pandas('jobmirror_db', 'online_bronze_job_descriptions')
    resume = read_bronze_table_as_pandas('jobmirror_db', 'online_bronze_labels')
    labels = read_bronze_table_as_pandas('jobmirror_db', 'online_bronze_resumes')
    jd.to_csv('jd_new.csv', index=False)
    resume.to_csv('resume_new.csv', index=False)
    labels.to_csv('labels_new.csv', index=False)
    
    



    
