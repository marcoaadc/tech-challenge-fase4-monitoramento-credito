from __future__ import annotations

import json
import logging

import pytest

from creditwatch.observability.events import BLOCKED, FAILED, SUCCESS, StageBlocked, StageTracker, read_events
from creditwatch.observability.logs import LOG_FILENAME, configure_logging, current_run_id, new_run_id, set_run_id
from creditwatch.observability.prometheus import PipelineMetrics
from creditwatch.observability.tracking import Tracker, _clean_metrics, export_tracking_summary
from creditwatch.pipelines.run_all import export_run_log


def test_run_id_is_unique_and_propagated():
    first, second = new_run_id(), new_run_id()
    assert first != second
    assert current_run_id() == second
    set_run_id("fixo")
    assert current_run_id() == "fixo"


def test_json_log_file_carries_run_id_and_extras(tmp_path):
    configure_logging("INFO", tmp_path)
    run_id = new_run_id()
    logging.getLogger("creditwatch.test").info("lote validado", extra={"batch": "2025-01", "rows": 2000})
    logging.getLogger("creditwatch.test").debug("nao deve aparecer")
    try:
        raise ValueError("falha simulada")
    except ValueError:
        logging.getLogger("creditwatch.test").exception("erro no estagio")
    for handler in logging.getLogger().handlers:
        handler.flush()

    records = [json.loads(line) for line in (tmp_path / LOG_FILENAME).read_text(encoding="utf-8").splitlines()]
    assert [record["message"] for record in records] == ["lote validado", "erro no estagio"]
    assert records[0]["run_id"] == run_id
    assert records[0]["batch"] == "2025-01"
    assert records[0]["rows"] == 2000
    assert records[0]["level"] == "INFO"
    assert "ValueError: falha simulada" in records[1]["exception"]


def test_configure_logging_without_directory_only_uses_console(tmp_path):
    configure_logging("WARNING")
    assert len(logging.getLogger().handlers) == 1
    assert logging.getLogger().level == logging.WARNING
    assert not list(tmp_path.iterdir())


def test_stage_tracker_records_success_with_duration_and_rows(tmp_path):
    set_run_id("run-1")
    tracker = StageTracker("monitor", tmp_path / "obs" / "events.jsonl")
    with tracker.stage("score", batch="2025-01", rows_in=2000) as event:
        event.rows_out = 1990
        event.details = {"latency": 1.5}

    stored = read_events(tmp_path / "obs" / "events.jsonl")
    assert len(stored) == 1
    assert stored[0]["status"] == SUCCESS
    assert stored[0]["run_id"] == "run-1"
    assert (stored[0]["rows_in"], stored[0]["rows_out"]) == (2000, 1990)
    assert stored[0]["duration_ms"] >= 0
    assert stored[0]["cpu_ms"] >= 0  # tempo de CPU do estagio
    assert stored[0]["details"] == {"latency": 1.5}


def test_stage_tracker_absorbs_blocked_and_propagates_failures(tmp_path):
    tracker = StageTracker("monitor", tmp_path / "events.jsonl")
    with tracker.stage("contract", batch="2025-07-r1"):
        raise StageBlocked("contrato reprovado")
    with pytest.raises(RuntimeError, match="disco cheio"), tracker.stage("score", batch="2025-07-r2"):
        raise RuntimeError("disco cheio")
    with tracker.stage("drift"):
        pass

    assert [event.status for event in tracker.events] == [BLOCKED, FAILED, SUCCESS]
    assert tracker.events[0].error == "contrato reprovado"
    assert tracker.events[1].error == "RuntimeError: disco cheio"
    summary = tracker.summary()
    assert (summary["stages"], summary["success"], summary["blocked"], summary["failed"]) == (3, 1, 1, 1)
    assert len(read_events(tmp_path / "events.jsonl")) == 3


def test_stage_tracker_works_in_memory_and_reads_missing_file(tmp_path):
    tracker = StageTracker("ingest")
    with tracker.stage("load"):
        pass
    assert tracker.events[0].as_dict()["pipeline"] == "ingest"
    assert read_events(tmp_path / "missing.jsonl") == []


def _scored_summary() -> dict:
    return {
        "batch": "2025-08",
        "status": "scored",
        "rows_received": 2000,
        "rows_scored": 2000,
        "contract": {"violated_rules": []},
        "drift": {
            "n_drifted": 2,
            "share_drifted": 0.4,
            "features": [{"feature": "limit_bal", "psi": 0.31}, {"feature": "age", "psi": 0.02}],
        },
        "prediction": {"score_psi": 0.2, "approval_rate": 0.55},
        "performance": {"roc_auc": 0.74, "ks": 0.38, "default_rate": 0.35, "bad_rate_approved": 0.2},
        "alerts": [{"kind": "data_drift", "severity": "critical"}, {"kind": "performance", "severity": "warning"}],
    }


def test_pipeline_metrics_export_in_prometheus_format(tmp_path):
    metrics = PipelineMetrics()
    tracker = StageTracker("monitor", metrics=metrics)
    with tracker.stage("score", batch="2025-08"):
        pass
    with tracker.stage("contract", batch="2025-07-r1"):
        raise StageBlocked("bloqueado")
    metrics.observe_batch(_scored_summary(), "v1")
    blocked = {
        "batch": "2025-07",
        "status": "blocked",
        "rows_received": 2040,
        "rows_scored": 0,
        "contract": {"violated_rules": ["DQ-01", "DQ-09"]},
        "alerts": [{"kind": "data_quality", "severity": "critical"}],
    }
    metrics.observe_batch(blocked, "v1")
    text = metrics.export(tmp_path / "obs" / "metrics.prom").read_text(encoding="utf-8")

    assert 'creditwatch_stage_runs_total{stage="score",status="success"} 1.0' in text
    assert 'creditwatch_stage_runs_total{stage="contract",status="blocked"} 1.0' in text
    assert 'creditwatch_rows_total{outcome="scored"} 2000.0' in text
    assert 'creditwatch_rows_total{outcome="quarantined"} 2040.0' in text
    assert 'creditwatch_contract_violations_total{rule="DQ-09"} 1.0' in text
    assert 'creditwatch_feature_psi{feature="limit_bal"} 0.31' in text
    assert "creditwatch_model_roc_auc 0.74" in text
    assert 'creditwatch_alerts_total{kind="data_drift",severity="critical"} 1.0' in text
    assert 'creditwatch_last_batch_info{batch="2025-08",model_version="v1"} 1.0' in text
    assert "creditwatch_stage_duration_seconds_bucket" in text
    assert 'creditwatch_stage_cpu_seconds_total{stage="score"}' in text
    assert "_created" not in text


def test_pushgateway_failure_does_not_break_the_export(tmp_path):
    metrics = PipelineMetrics()
    path = metrics.export(tmp_path / "metrics.prom", pushgateway_url="http://127.0.0.1:9")  # porta fechada
    assert path.exists()


def test_tracker_is_a_noop_without_tracking_uri(tmp_path):
    tracker = Tracker(None, "experimento")
    assert not tracker.enabled
    with tracker.run("execucao") as active:
        assert active is None
        tracker.log_params({"a": 1})
        tracker.log_metrics({"auc": 0.7}, step=1)
        tracker.set_tags({"fase": "teste"})
        tracker.log_artifact(tmp_path / "inexistente.json")
        tracker.log_dict({"a": 1}, "a.json")
        tracker.log_model(object(), None)


def test_clean_metrics_drops_non_finite_and_non_numeric_values():
    cleaned = _clean_metrics(
        {"roc_auc": 0.7, "nan": float("nan"), "inf": float("inf"), "text": "2026-01", "flag": True, "psi/feature@1": 1}
    )
    assert cleaned == {"roc_auc": 0.7, "flag": 1.0, "psi/feature_1": 1.0}


def test_tracking_summary_is_skipped_without_mlflow(tmp_path):
    assert export_tracking_summary(None, ("experimento",), tmp_path / "summary.json") is None
    assert not (tmp_path / "summary.json").exists()


def test_run_log_export_keeps_only_the_lines_of_the_run(isolated_settings):
    settings = isolated_settings
    assert export_run_log(settings, "run-a") is None  # ainda nao ha log

    configure_logging("INFO", settings.logs_dir)
    set_run_id("run-a")
    logging.getLogger("creditwatch.test").info("primeira execucao")
    set_run_id("run-b")
    logging.getLogger("creditwatch.test").info("segunda execucao")
    set_run_id("run-a")
    logging.getLogger("creditwatch.test").info("primeira execucao, de novo")

    exported = export_run_log(settings, "run-a")
    records = [json.loads(line) for line in exported.read_text(encoding="utf-8").splitlines()]
    assert [record["message"] for record in records] == ["primeira execucao", "primeira execucao, de novo"]
    assert {record["run_id"] for record in records} == {"run-a"}
    assert exported == settings.observability_dir / "pipeline_log.jsonl"
