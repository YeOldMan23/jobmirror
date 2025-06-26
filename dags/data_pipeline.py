from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.models import Variable, XCom
from airflow.hooks.base import BaseHook

# included a dummy reference
# from src.monitoring import monitoring

from datetime import datetime, timedelta


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1, # retry once evry 5 minutes
    'retry_delay': timedelta(minutes=5),
}

dag =  DAG(
    'data-pipeline',
    default_args=default_args,
    params={
        "bronze_start": 0,
        "bronze_end": 100, # feed in data in batches
    },
    description='data pipeline run once a month',
    schedule_interval='0 0 1 * *',  # At 00:00 on day-of-month 1: when you want to run (translate to cron)
    start_date=datetime(2022, 9, 1), 
    end_date=datetime(2022, 12, 1), # setting it to the max date we have
    catchup=False,
    max_active_runs=1,

)

###########################
###### Data pipeline ######
###########################

#### Data preprocessing ####

###### Bronze Table ######
# Bronze tables processing includes label and features 

# Retrieves data from huggingface, extract text using LLM and parse to MongoDB
bronze_store = BashOperator(
    task_id='run_bronze_feature_and_label_store',
    bash_command=(
        'cd /opt/airflow/utils && '
        'python3 data_processing_bronze_table.py '
        '--start {{ params.bronze_start }} '
        '--end {{ params.bronze_end }} '
        '--batch_size 10 '
    ),
    dag=dag
)

###### Silver Label Table ######   
silver_label_store = BashOperator(
    task_id='run_silver_label_store',
    bash_command=(
    'cd /opt/airflow/utils && '
    'python3 data_processing_silver_table.py '
    '--snapshotdate "{{ ds }}" '
    '--task data_processing_silver_labels '
    ),
    dag=dag
)

###### Silver Feature Tables ###### 
# Processing for resume silver table
silver_resume_store = BashOperator(
    task_id='run_silver_resume_store',
    bash_command=(
    'cd /opt/airflow/utils && '
    'python3 data_processing_silver_table.py '
    '--snapshotdate "{{ ds }}" '
    '--task data_processing_silver_resume '
    ),
    dag=dag
)

# Processing for jd silver table
silver_jd_store = BashOperator(
    task_id='run_silver_jd_store',
    bash_command=('cd /opt/airflow/utils && '
    'python3 data_processing_silver_table.py '
    '--snapshotdate "{{ ds }}" '
    '--task data_processing_silver_jd '
    ),
    dag=dag
)

# Merge silver resume and silver JD tables into combined silver table
silver_combined = BashOperator(
    task_id='run_silver_combined',
    bash_command=('cd /opt/airflow/utils && '
    'python3 data_processing_silver_table.py '
    '--snapshotdate "{{ ds }}" '
    '--task data_processing_silver_combined '
    ),
    dag=dag
)

###### Gold Tables ######
gold_feature_store = BashOperator(
    task_id='run_gold_feature_store',
    bash_command=(
        'cd /opt/airflow/utils && '
        'python3 data_processing_gold_table.py '
        '--snapshotdate "{{ ds }}" '
        '--store feature '
    ),
    dag=dag
)
gold_label_store = BashOperator(
    task_id='run_gold_label_store',
    bash_command=(
        'cd /opt/airflow/utils && '
        'python3 data_processing_gold_table.py '
        '--snapshotdate "{{ ds }}" '
        '--store label '
    ),
    dag=dag
)

# Define task dependencies to run scripts sequentially
bronze_store >> [silver_resume_store, silver_jd_store, silver_label_store] >> silver_combined
silver_combined >> [gold_feature_store, gold_label_store]
