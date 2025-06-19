"""
Test the silver table stuff 
"""
from utils.data_processing_silver_table import *
from utils.date_utils import *
from utils.mongodb_utils import get_pyspark_session
import os
from utils.gdrive_utils import get_folder_id_by_path, connect_to_gdrive
# from utils.edu_silver_transform import resume_edu_to_silver


if __name__ == "__main__":
    # Get the pyspark session
    spark = get_pyspark_session()
    
    # Datamart dir
    datamart_dir = os.path.join(os.getcwd(), "datamart")

    # Get the range of dates
    date_range = get_snapshot_dates(datetime(2021, 1, 1), datetime(2021, 12, 30))

    # For each range, read the silver table and parse
    for cur_date in date_range:
        snapshot_date = f"{cur_date.year}-{cur_date.month}"

        print("Processing silver {}".format(snapshot_date))

        data_processing_silver_resume(cur_date, spark)
        data_processing_silver_jd(cur_date, spark)
        data_processing_silver_labels(cur_date, spark)
        data_processing_silver_combined(cur_date, spark)
        # resume_edu_to_silver(datamart_dir, snapshot_date, spark)            # testing for education


