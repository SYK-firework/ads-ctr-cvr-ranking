"""Retraining decision used by the Airflow DAG, and the DAG's structure (when Airflow is installed)."""
from pathlib import Path

import pytest

from adsrank.pipeline import retrain_decision


def _w(**kw):
    base = {"day": 0, "hours": 0.0, "n": 50_000, "psi_p_ctr": 0.002, "psi_p_cvr": 0.001,
            "psi_campaign": 0.02, "calib_p_ctr": 0.99, "calib_p_cvr": 1.01}
    return base | kw


def test_no_retrain_when_stable():
    assert retrain_decision({"daily": [_w(), _w(day=1)]})["retrain"] is False


def test_each_trigger():
    assert retrain_decision({"daily": [_w(psi_p_ctr=0.3)]})["retrain"]
    assert retrain_decision({"daily": [_w(psi_campaign=0.4)]})["retrain"]
    assert retrain_decision({"daily": [_w(calib_p_ctr=0.93)]})["retrain"]
    assert retrain_decision({"daily": [_w()]}, force=True)["reasons"] == ["forced"]


def test_small_windows_do_not_trigger_on_mix_or_calibration():
    small = _w(n=9_896, psi_campaign=0.364, calib_p_ctr=0.942, calib_p_cvr=0.889)
    assert retrain_decision({"daily": [small]})["retrain"] is False
    assert retrain_decision({"daily": [_w(n=9_896, psi_p_ctr=0.5)]})["retrain"]   # score PSI always counts


def test_dag_structure():
    pytest.importorskip("airflow")
    from airflow.models import DagBag
    bag = DagBag(dag_folder=str(Path(__file__).parents[1] / "dags"))   # examples off via env in CI
    assert not bag.import_errors, bag.import_errors
    dag = bag.dags["adsrank_retrain"]
    t = {x.task_id: x for x in dag.tasks}
    assert set(t) == {"download_data", "calibrate", "drift_report", "decide", "skip_retrain",
                      "export_with_skew_check", "build_image", "canary_rollout"}
    assert t["decide"].downstream_task_ids == {"skip_retrain", "export_with_skew_check"}
    assert t["build_image"].upstream_task_ids == {"export_with_skew_check"}
    assert t["canary_rollout"].upstream_task_ids == {"build_image"}
