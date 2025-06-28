from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.models import Variable, XCom
from airflow.hooks.base import BaseHook

# included a dummy reference
# from src.monitoring import monitoring

from datetime import datetime, timedelta

# try:
#     Variable.get("processing_type")
# except:
#     Variable.set("processing_type", "inference")

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
    # params={
    #     "bronze_start": 0,
    #     "bronze_end": 2,
    # },
    description='inference pipeline run once a month',
    schedule_interval='0 0 1 * *',  # At 00:00 on day-of-month 1: when you want to run (translate to cron)
    start_date=datetime(2022, 9, 1), 
    end_date=datetime(2022, 10, 1), 
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
# Retrieves processed gold features

###### Gold Tables ######
get_gold_features = DummyOperator(task_id="fetch_processed_features_for_training", dag=dag)
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

# model inferencing
model_inference_start = DummyOperator(task_id="model_inference_start", dag=dag)
inference_task = BashOperator(
    task_id='make_predictions',
    bash_command = 'python /opt/model_deploy/inference.py',
    dag=dag
)


# --- model monitoring ---
model_monitor_start = DummyOperator(task_id="model_monitor_start", dag=dag)

monitoring_task_prod = BashOperator(
    task_id='monitoring_prod',
    bash_command = 'python /opt/model_monitor/model_monitoring_prod.py',
    dag=dag
)

monitoring_task_shad = BashOperator(
    task_id='monitoring_shad',
    bash_command = 'python /opt/model_monitor/model_monitoring_shad.py',
    dag=dag
)


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
model_monitor_production = BashOperator(task_id="model_monitoring",
                               bash_command = 'python /opt/utils/model_monitoring.py',
                               dag=dag)

model_monitor_shadow = BashOperator(task_id="model_monitoring",
                               bash_command = 'python /opt/utils/model_monitoring.py',
                               dag=dag)

# Define task dependencies to run scripts sequentially
# model_inference_completed >> model_monitor_start
get_gold_features >> [gold_feature_store, gold_label_store]
[gold_feature_store, gold_label_store] >> model_inference_start
model_inference_start >> inference_task >> model_monitor_start
model_monitor_start >>[ model_monitor_production,model_monitor_shadow] >> model_monitor_completed

