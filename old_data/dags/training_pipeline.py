from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1, # retry once evry 5 minutes
    'retry_delay': timedelta(minutes=5),
    
}

with DAG(
    'training-pipeline',
    default_args=default_args,
    description='training pipeline run once a month',
    schedule_interval='0 0 1 * *',  # At 00:00 on day-of-month 1: when you want to run (translate to cron)
    start_date=datetime(2021, 9, 1), # 6 has no data , 2021, 8, 1 training on 3 months of data 
    end_date=datetime(2021, 11, 1),
    catchup=True,
    tags=['training']
) as dag:
    
    get_gold_features = DummyOperator(task_id="fetch_processed_features_for_training", dag=dag)

    train_xgb = BashOperator(
        task_id='train_xgboost_classifier',
        bash_command='PYTHONPATH=/opt/airflow python /opt/airflow/model_train/train_xgb.py --snapshotdate "{{ ds }}"',
    )

    train_lgr = BashOperator(
        task_id='train_logistic_regression_classifier',
        bash_command='PYTHONPATH=/opt/airflow python /opt/airflow/model_train/train_logreg.py --snapshotdate "{{ ds }}" ',
    )

    promote_best = BashOperator(
        task_id='promote_best_model',
        bash_command='PYTHONPATH=/opt/airflow python /opt/airflow/model_train/promote_best.py --snapshotdate "{{ ds }}" ',
    )

    [train_xgb, train_lgr] >> promote_best