"""Daily drift check -> (if needed) retrain -> skew-checked export -> image -> gated canary.

    download_data >> calibrate >> drift_report >> decide
    decide -> skip_retrain                                              (no drift: done)
    decide -> export_with_skew_check >> build_image >> canary_rollout   (drift: retrain and roll out)

Each step is a `make` target of this repository, so the DAG is the same pipeline you run by hand.
`decide` reads results/drift.json and applies adsrank.pipeline.retrain_decision; the canary step
promotes the new image only if every traffic step passes the gate, and otherwise rolls back and
fails the task, which leaves stable untouched.

Setup: put this file in $AIRFLOW_HOME/dags (or point dags_folder here) and set ADSRANK_HOME to the
repository path. The canary step needs kubectl / kind access to the cluster from the worker.
Trigger with {"force": true} to retrain regardless of drift.
"""
import json
import os
from datetime import datetime, timedelta

from airflow import DAG

try:                                                    # Airflow 3
    from airflow.providers.standard.operators.bash import BashOperator
    from airflow.providers.standard.operators.empty import EmptyOperator
    from airflow.providers.standard.operators.python import BranchPythonOperator
except ImportError:                                     # Airflow 2
    from airflow.operators.bash import BashOperator
    from airflow.operators.empty import EmptyOperator
    from airflow.operators.python import BranchPythonOperator

HOME = os.environ.get("ADSRANK_HOME", os.path.expanduser("~/ads-ctr-cvr-ranking"))
ROWS = os.environ.get("ADSRANK_ROWS", "2000000")
IMAGE = "adsrank-serving:{{ ts_nodash | lower }}"


def decide(**context):
    from adsrank.pipeline import retrain_decision
    report = json.loads(open(os.path.join(HOME, "results", "drift.json")).read())
    force = bool((context.get("params") or {}).get("force", False))
    d = retrain_decision(report, force=force)
    print(json.dumps(d, indent=1))
    return "export_with_skew_check" if d["retrain"] else "skip_retrain"


with DAG(
    dag_id="adsrank_retrain",
    description="Drift check, then retrain + canary when drift is found",
    schedule="@daily",
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    params={"force": False},
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["ads", "ranking", "mlops"],
) as dag:
    def make(task_id, target, timeout_min=60):
        return BashOperator(task_id=task_id, cwd=HOME, bash_command=f"make {target}",
                            execution_timeout=timedelta(minutes=timeout_min))

    download = make("download_data", f"data-attribution ATTR_ROWS={ROWS}")
    calibrate = make("calibrate", f"calibrate ATTR_ROWS={ROWS}")
    drift = make("drift_report", "drift", 15)
    branch = BranchPythonOperator(task_id="decide", python_callable=decide)
    skip = EmptyOperator(task_id="skip_retrain")
    export = make("export_with_skew_check", f"export ATTR_ROWS={ROWS}")          # fails on skew
    image = make("build_image", f"canary-image CANARY_IMAGE={IMAGE}", 30)
    canary = make("canary_rollout", f"canary CANARY_IMAGE={IMAGE}", 30)           # fails on rollback

    download >> calibrate >> drift >> branch
    branch >> skip
    branch >> export >> image >> canary
