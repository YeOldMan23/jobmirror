"""
Test the silver table stuff 
"""
from utils.src.data_processing_silver_table import *
from utils.date_utils import *
from utils.mongodb_utils import get_pyspark_session
import os
from utils.gdrive_utils import get_folder_id_by_path, connect_to_gdrive
# from utils.edu_silver_transform import resume_edu_to_silver


if __name__ == "__main__":

    # Get the pyspark session
    spark = get_pyspark_session()    
    
    # Datamart dir
    # datamart_dir = os.path.join(os.getcwd(), "datamart/")

    # Get the range of dates
    date_range = get_snapshot_dates(datetime(2022, 6, 1), datetime(2022, 12, 1))
    print(date_range)
    # For each range, read the silver table and parse
    for cur_date in date_range:
        snapshot_date = f"{cur_date.year}-{cur_date.month:02d}"
        print("Processing silver {}".format(snapshot_date))
        data_processing_silver_resume(cur_date, "inference", spark)
        data_processing_silver_jd(cur_date, "inference", spark)
        data_processing_silver_labels(cur_date, "inference", spark)
        data_processing_silver_combined(cur_date, "inference", spark)
        # resume_edu_to_silver(datamart_dir, snapshot_date, spark)            # testing for education


