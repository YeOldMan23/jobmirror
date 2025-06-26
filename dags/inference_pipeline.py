from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.models import Variable, XCom
from airflow.hooks.base import BaseHook

# included a dummy reference
# from src.monitoring import monitoring

from datetime import datetime, timedelta

try:
    Variable.get("processing_type")
except:
    Variable.set("processing_type", "inference")

# # this is assuming an auto training. else we automatically set whether to train or not. 

# def check_monitoring_results(**context):
#     """
#     Check monitoring results and set processing type
#     Returns 'training' if issues detected, otherwise 'inference' to put in BashOperator
#     """
#     # Get monitoring results from XCom
#     ti = context['ti']
#     model1_issues = ti.xcom_pull(task_ids='model_monitor_start', key='fail')
    
#     # Determine type based on monitoring results
#     processing_type = "training" if model1_issues else "inference"
    
#     # Set the Variable for other tasks to use
#     Variable.set("processing_type", processing_type)
    
#     print(f"Setting processing type to: {processing_type}")
#     return processing_type

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1, # retry once evry 5 minutes
    'retry_delay': timedelta(minutes=5),
}

dag =  DAG(
    'ml-pipeline',
    default_args=default_args,
    params={
        "bronze_start": 0,
        "bronze_end": 2,
    },
    description='data pipeline run once a month',
    schedule_interval='0 0 1 * *',  # At 00:00 on day-of-month 1: when you want to run (translate to cron)
    start_date=datetime(2022, 9, 1), 
    # end_date=datetime(2021, 12, 1),
    catchup=False,
    max_active_runs=1,
    tags=['inference']
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
        '--type {{ var.value.processing_type }}'
    ),
    dag=dag
)

###### Silver Label Table ######   
# silver_label_store = DummyOperator(task_id="silver_label_store")
silver_label_store = BashOperator(
    task_id='run_silver_label_store',
    bash_command='cd /opt/airflow/utils && '
    'python3 data_processing_silver_table.py '
    '--snapshotdate "2022-09-01" '
    '--task data_processing_silver_labels '
    '--type {{ var.value.processing_type }}',
    dag=dag
)

###### Silver Feature Tables ###### 
# Processing for resume silver table
# silver_table_1 = DummyOperator(task_id="silver_table_1")
silver_resume_store = BashOperator(
    task_id='run_silver_resume_store',
    bash_command='cd /opt/airflow/utils && '
    'python3 data_processing_silver_table.py '
    '--snapshotdate "2022-09-01" '
    '--task data_processing_silver_resume '
    '--type {{ var.value.processing_type }}',
    dag=dag
)

# Processing for jd silver table
silver_jd_store = BashOperator(
    task_id='run_silver_jd_store',
    bash_command='cd /opt/airflow/utils && '
    'python3 data_processing_silver_table.py '
    '--snapshotdate "2022-09-01" '
    '--task data_processing_silver_jd '
    '--type {{ var.value.processing_type }}',
    dag=dag
)

# Merge silver resume and silver JD tables into combined silver table
silver_combined = BashOperator(
    task_id='run_silver_combined',
    bash_command='cd /opt/airflow/utils && '
    'python3 data_processing_silver_table.py '
    '--snapshotdate "2022-09-01" '
    '--task data_processing_silver_combined '
    '--type {{ var.value.processing_type }}',
    dag=dag
)

###### Gold Tables ######
gold_feature_store = BashOperator(
    task_id='run_gold_feature_store',
    bash_command=(
        'cd /opt/airflow/utils && '
        'python3 data_processing_gold_table.py '
        '--snapshotdate "2022-09-01" '
        '--type {{ var.value.processing_type }} '
        '--store feature '
    ),
    dag=dag
)
gold_label_store = BashOperator(
    task_id='run_gold_label_store',
    bash_command=(
        'cd /opt/airflow/utils && '
        'python3 data_processing_gold_table.py '
        '--snapshotdate "2022-09-01" '
        '--type {{ var.value.processing_type }} '
        '--store label '
    ),
    dag=dag
)

# model inferencing
model_inference_start = DummyOperator(task_id="model_inference_start", dag=dag)
inference_task = BashOperator(
    task_id='make_predictions',
    bash_command = 'python /opt/model_deploy/load_model.py',
    dag=dag
)

## model training 
# train_logreg = BashOperator(
#     task_id='train_logistic_regression',
#     bash_command='python /opt/model_train/train_logreg.py',
#     dag=dag
# )

# train_xgb = BashOperator(
#     task_id='train_xgboost_classifier',
#     bash_command='python /opt/model_train/train_xgb.py',
#     dag=dag
# )

# promote = BashOperator(
#     task_id='promote_best_model',
#     bash_command='python /opt/model_train/promote_best.py',
#     trigger_rule='one_success',
#     dag=dag
# )

# deploy = BashOperator(
#     task_id='model_deploy',
#     bash_command='python /opt/model_deploy/model_deploy.py',
#     dag=dag
# )

# --- model monitoring ---
model_monitor_start = DummyOperator(task_id="model_monitor_start", dag=dag)
# model_monitor_start = PythonOperator(
#     task_id="model_monitor_start",
#     python_callable=model_monitoring,
#     dag=dag)

# assuming we push the xcom keys as 'fail' it will set --type as training
# model_1_monitor = PythonOperator(
#     task_id="model_1_monitor",
#     python_callable=check_monitoring_results,
#     dag=dag)

model_monitor_completed = DummyOperator(task_id="model_monitor_completed",dag=dag)
model_monitor = BashOperator(task_id="model_monitoring",
                               bash_command = 'python /opt/utils/model_monitoring.py',
                               dag=dag)

# Define task dependencies to run scripts sequentially
# model_inference_completed >> model_monitor_start
bronze_store >> [silver_resume_store, silver_jd_store, silver_label_store] >> silver_combined
silver_combined >> [gold_feature_store, gold_label_store]
[gold_feature_store, gold_label_store] >> model_inference_start
model_inference_start >> inference_task >> model_monitor_start
model_monitor_start >> model_monitor >> model_monitor_completed


# --- model auto training ---
# model_automl_start = DummyOperator(task_id="model_automl_start", dag=dag)

# model_1_automl = DummyOperator(task_id="model_1_automl", dag=dag)

# model_2_automl = DummyOperator(task_id="model_2_automl", dag=dag)

# model_automl_completed = DummyOperator(task_id="model_automl_completed", dag=dag)

# Define task dependencies to run scripts sequentially
# feature_store_completed >> model_automl_start
# label_store_completed >> model_automl_start
# model_automl_start >> model_1_automl >> model_automl_completed
# model_automl_start >> model_2_automl >> model_automl_completed