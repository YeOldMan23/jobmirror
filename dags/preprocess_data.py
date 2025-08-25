from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.models import Variable, XCom
from airflow.hooks.base import BaseHook


from datetime import datetime, timedelta


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1, # retry once evry 5 minutes
    'retry_delay': timedelta(minutes=5),
}

dag =  DAG(
    'preprocess-data',
    default_args=default_args,
    params={
        "bronze_start": 0,
        "bronze_end": 100, # feed in data in batches
        "bronze_skip": True
    },
    description='Download and process the data, run once a month',
    schedule_interval='0 0 1 * *',  # At 00:00 on day-of-month 1: when you want to run (translate to cron)
    start_date=datetime(2022, 9, 1), 
    end_date=datetime(2022, 12, 1), # setting it to the max date we have
    catchup=True,
    max_active_runs=1, # My computer sucks

)

download_model = BashOperator(
    task_id = "download_llm_model"
    bash_command = (
        'cd /opt/airflow/llm && '
        'python3 download_llama.py'
    )
    dag = dag
)