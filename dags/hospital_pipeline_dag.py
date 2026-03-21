import json
import os
from datetime import datetime,timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "ritam",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

PROJECT_DIR = "/opt/airflow/project"

with DAG(
    dag_id="hospital_readmission_pipeline",
    default_args=default_args,
    description=(
        "Fetches new 10K patient records from API, "
        "replaces old data in PostgreSQL, then runs "
        "full ML pipeline via dvc repro."
    ),
    schedule="@weekly",
    start_date=datetime(2026, 3, 21),
    catchup=False,
    tags=["hospital", "readmission", "ml-pipeline"],
) as dag:

    fetch_new_data = BashOperator(
        task_id="fetch_new_data",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            f"python -m src.data.etl.pipeline --full 10000 1000 --replace"
        ),
        doc_md=(
            "Fetches next 10K records from API in 10 loops of 1000 records each. "
            "Uses sliding window offset from state.json. "
            "Replaces old PostgreSQL records. Advances state.json offset by 10K."
        ),
    )

    dvc_repro = BashOperator(
        task_id="dvc_repro",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            f"dvc repro"
        ),
        doc_md=(
            "Runs full ML pipeline via dvc repro. "
            "DVC skips unchanged stages automatically."
        ),
        execution_timeout=timedelta(minutes=60),
    )

    def print_pipeline_summary(**context):
        print("=" * 60)
        print("HOSPITAL READMISSION PIPELINE - RUN COMPLETE")
        print("=" * 60)
 
        # ETL state — what offset did we advance to?
        state_path = os.path.join(
            PROJECT_DIR, "data", "etl_state", "state.json"
        )
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                state = json.load(f)
            print(f"Current data offset : {state.get('last_offset', 'N/A')}")
        else:
            print("ETL state file not found")
 
        # Drift decision — was drift detected?
        drift_path = os.path.join(
            PROJECT_DIR, "models", "drift_decision.json"
        )
        if os.path.exists(drift_path):
            with open(drift_path, "r") as f:
                drift = json.load(f)
            print(f"Drift detected      : {drift.get('drift_detected', 'N/A')}")
            drift_share = drift.get("drift_share")
            if drift_share is not None:
                print(f"Drift share         : {drift_share * 100:.1f}%")
        else:
            print("Drift decision file not found")
 
        # Deployment decision — was new model deployed?
        deploy_path = os.path.join(
            PROJECT_DIR, "models", "deployment_decision.json"
        )
        if os.path.exists(deploy_path):
            with open(deploy_path, "r") as f:
                decision = json.load(f)
            print(f"Model deployed      : {decision.get('deploy', 'N/A')}")
            print(f"Reason              : {decision.get('reason', 'N/A')}")
            cand = decision.get("cand_metrics", {})
            if cand:
                print(f"Candidate ROC-AUC   : {cand.get('roc_auc', 'N/A'):.4f}")
                print(f"Candidate F1        : {cand.get('f1', 'N/A'):.4f}")
            prod = decision.get("prod_metrics", {})
            if prod:
                print(f"Production ROC-AUC  : {prod.get('roc_auc', 'N/A'):.4f}")
        else:
            print("Deployment decision file not found")
 
        print("=" * 60)
 
    pipeline_summary = PythonOperator(
        task_id="pipeline_summary",
        python_callable=print_pipeline_summary,
        # ALL_DONE means this runs even if dvc_repro fails
        # so you always get a summary in the logs
        trigger_rule="all_done",
        doc_md="Prints run summary — always executes regardless of upstream status.",
    )