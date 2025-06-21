from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from airflow.models import Variable

from utils import data_processing_bronze_table, data_processing_silver_table, data_processing_gold_table

# Set batch indices for bronze table loading
Variable.set("bronze_start_index", 0)
Variable.set("bronze_end_index", 1000)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1, # retry once evry 5 minutes
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'dag',
    default_args=default_args,
    description='data pipeline run once a month',
    schedule_interval='0 0 1 * *',  # At 00:00 on day-of-month 1: when you want to run (translate to cron)
    start_date=datetime(2023, 1, 1), 
    end_date=datetime(2024, 12, 1),
    catchup=True,
) as dag:

###########################
###### Data pipeline ######
###########################

###### Bronze Table ######
# Bronze tables processing includes label and features 
# Retrieves data from huggingface, extract text using LLM and parse to MongoDB
    bronze_store = BashOperator(
        task_id='run_bronze_feature_and_label_store',
        bash_command=(
            'cd /opt/airflow && '
            'python3 data-processing.py '
            '--start {{ var.value.bronze_start_index }} '
            '--end {{ var.value.bronze_end_index }} '
            '--batch_size 100 '
        ),
    )

###### Silver Label Table ######   
 # Input dummy operator for testing purpose. Actual script commented below.
    silver_label_store = DummyOperator(task_id="silver_label_store")
    silver_label_store = BashOperator(
    task_id='run_silver_jd_store',
    bash_command='cd /opt/airflow/scripts && '
    'python3 data_processing_silver_table.py '
    '--snapshotdate "{{ ds }}" '
    '--task data_processing_silver_labels',
    dag=dag
    )

###### Gold Table ######
# Input dummy operator for testing purpose. Actual script commented below.
    gold_label_store = DummyOperator(task_id="gold_label_store")
    # gold_label_store = BashOperator(
    #     task_id='run_gold_label_store',
    #     bash_command=(
    #         'cd /opt/airflow/scripts && '
    #         'python3 gold_label_store.py '
    #         '--snapshotdate "{{ ds }}"'
    #     ),
    # )

    label_store_completed = DummyOperator(task_id="label_store_completed")

    # Define task dependencies to run scripts sequentially
    bronze_store >> silver_label_store >> gold_label_store >> label_store_completed
  
 # --- feature store --- chaining multiple bronze table to silver table
    # bronze_table_1 = DummyOperator(task_id="bronze_table_1")

    # Processing for resume silver table
    silver_table_1 = DummyOperator(task_id="silver_table_1")
    silver_resume_store = BashOperator(
    task_id='run_silver_resume_store',
    bash_command='cd /opt/airflow/scripts && '
    'python3 data_processing_silver_table.py '
    '--snapshotdate "{{ ds }}" '
    '--task data_processing_silver_resume',
    dag=dag
    )

    # Processing for jd silver table
    silver_jd_store = BashOperator(
    task_id='run_silver_jd_store',
    bash_command='cd /opt/airflow/scripts && '
    'python3 data_processing_silver_table.py '
    '--snapshotdate "{{ ds }}" '
    '--task data_processing_silver_jd',
    dag=dag
    )

    # Merge silver resume and silver JD tables into combined silver table
    silver_combined = BashOperator(
    task_id='run_silver_combined',
    bash_command='cd /opt/airflow/scripts && '
    'python3 data_processing_silver_table.py '
    '--snapshotdate "{{ ds }}" '
    '--task data_processing_silver_combined',
    dag=dag
    )

    [silver_resume_store, silver_jd_store] >> silver_combined

    # Create gold feature store
    gold_feature_store = DummyOperator(task_id="gold_feature_store")

    feature_store_completed = DummyOperator(task_id="feature_store_completed")
    
    # Define task dependencies to run scripts sequentially
    # Since bronze feature store is already created, we skip the task in this step
    bronze_store >> []
    # silver_table_1 >> big_silver_table >> gold_feature_store
    gold_feature_store >> feature_store_completed
    
    
    ## model training 
    train_logreg = BashOperator(
        task_id='train_logistic_regression',
        bash_command='python /opt/model_train/train_logreg.py',
    )

    train_xgb = BashOperator(
        task_id='train_xgboost_classifier',
        bash_command='python /opt/model_train/train_xgb.py',
    )

    promote = BashOperator(
        task_id='promote_best_model',
        bash_command='python /opt/model_train/promote_best.py',
    )
    
    deploy = BashOperator(
        task_id='model_deploy',
        bash_command='python /opt/model_deploy/model_deploy.py',
    )

    [train_logreg, train_xgb] >> promote>> deploy


    # --- model inference ---
    model_inference_start = DummyOperator(task_id="model_inference_start")

    model_1_inference = DummyOperator(task_id="model_1_inference")

    model_2_inference = DummyOperator(task_id="model_2_inference")

    model_inference_completed = DummyOperator(task_id="model_inference_completed")
    
    # Define task dependencies to run scripts sequentially
    feature_store_completed >> model_inference_start
    model_inference_start >> model_1_inference >> model_inference_completed
    model_inference_start >> model_2_inference >> model_inference_completed


    # --- model monitoring ---
    model_monitor_start = DummyOperator(task_id="model_monitor_start")

    model_1_monitor = DummyOperator(task_id="model_1_monitor")

    model_2_monitor = DummyOperator(task_id="model_2_monitor")

    model_monitor_completed = DummyOperator(task_id="model_monitor_completed")
    
    # Define task dependencies to run scripts sequentially
    model_inference_completed >> model_monitor_start
    model_monitor_start >> model_1_monitor >> model_monitor_completed
    model_monitor_start >> model_2_monitor >> model_monitor_completed


    # --- model auto training ---

    model_automl_start = DummyOperator(task_id="model_automl_start")
    
    model_1_automl = DummyOperator(task_id="model_1_automl")

    model_2_automl = DummyOperator(task_id="model_2_automl")

    model_automl_completed = DummyOperator(task_id="model_automl_completed")
    
    # Define task dependencies to run scripts sequentially
    feature_store_completed >> model_automl_start
    label_store_completed >> model_automl_start
    model_automl_start >> model_1_automl >> model_automl_completed
    model_automl_start >> model_2_automl >> model_automl_completed