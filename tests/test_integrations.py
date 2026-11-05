"""Testes das integracoes externas reais: Evidently (relatorio HTML) e MLflow (rastreamento)."""

from __future__ import annotations

import pytest

from creditwatch.data.schema import SCORE, TARGET
from creditwatch.data.split import SPLIT_COLUMN
from creditwatch.drift import evidently_report
from creditwatch.drift.detector import detect_drift
from creditwatch.features.engineering import add_engineered_features, model_matrix
from creditwatch.model.training import build_candidate
from creditwatch.observability.tracking import Tracker
from creditwatch.simulation.scenario import default_scenario, generate_batch


def _with_score(frame):
    """Score sintetico deterministico: o teste e do relatorio, nao do modelo."""
    enriched = add_engineered_features(frame)
    enriched[SCORE] = (0.15 + 0.1 * enriched["max_delay"]).clip(upper=0.95)
    return enriched


@pytest.mark.evidently
@pytest.mark.slow
def test_evidently_report_agrees_with_the_detector_and_saves_html(reference, pool, tmp_path):
    holdout = _with_score(reference[reference[SPLIT_COLUMN] != "train"].head(3000))
    drifted = _with_score(generate_batch(pool, default_scenario()[-1], 1500))

    snapshot = evidently_report.run_evidently_report(
        holdout, drifted, psi_threshold=0.10, drift_share=0.30, decision_threshold=0.3, batch="2025-12"
    )
    summary = evidently_report.summarize(snapshot, psi_threshold=0.10)
    own = detect_drift(holdout, drifted)

    assert summary["method"] == "psi"
    assert summary["n_columns"] == len(own.features) + 1  # as features monitoradas mais o score
    assert summary["share_drifted"] >= 0.30
    # Contraprova: as features criticas do detector proprio tambem sao apontadas pelo Evidently.
    assert set(own.critical_features) <= set(summary["drifted_columns"])
    assert "utilization_1" in summary["drifted_columns"]
    assert "sex" not in summary["drifted_columns"]

    html = evidently_report.save_html(snapshot, tmp_path / "evidently" / "drift.html")
    assert html.stat().st_size > 500_000
    assert "utilization_1" in html.read_text(encoding="utf-8")


@pytest.mark.evidently
@pytest.mark.slow
def test_evidently_report_without_labels_skips_classification_quality(reference, pool):
    holdout = _with_score(reference.head(1500))
    stable = _with_score(generate_batch(pool, default_scenario()[0], 1000)).drop(columns=[TARGET])
    snapshot = evidently_report.run_evidently_report(holdout, stable, decision_threshold=0.3)
    summary = evidently_report.summarize(snapshot)
    assert summary["n_drifted"] == 0
    assert not any("RocAuc" in metric["metric_name"] for metric in snapshot.dict()["metrics"])


@pytest.mark.slow
def test_mlflow_tracker_logs_runs_metrics_and_artifacts(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.chdir(tmp_path)  # o diretorio de artefatos padrao do MLflow e relativo ao cwd
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    tracker = Tracker(uri, "creditwatch-teste")
    assert tracker.enabled

    artifact = tmp_path / "summary.json"
    artifact.write_text("{}", encoding="utf-8")
    with tracker.run("monitoramento", tags={"stage": "monitoring"}) as parent:
        tracker.log_params({"model_version": "v1", "psi_critical": 0.25})
        for step, auc in enumerate([0.78, 0.74, 0.61]):
            tracker.log_metrics({"performance/roc_auc": auc, "ignorado": float("nan")}, step=step)
        with tracker.run("2025-12", nested=True, tags={"status": "scored"}) as child:
            tracker.log_metrics({"drift/n_drifted": 19})
            tracker.log_artifact(artifact)
            tracker.set_tags({"phase": "drift_de_conceito"})
            tracker.log_dict({"a": 1}, "payload.json")

    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    history = client.get_metric_history(parent.info.run_id, "performance/roc_auc")
    assert [(point.step, point.value) for point in history] == [(0, 0.78), (1, 0.74), (2, 0.61)]
    assert client.get_run(parent.info.run_id).data.params["model_version"] == "v1"
    child_run = client.get_run(child.info.run_id)
    assert child_run.data.tags["mlflow.parentRunId"] == parent.info.run_id
    assert child_run.data.tags["phase"] == "drift_de_conceito"
    assert {item.path for item in client.list_artifacts(child.info.run_id)} == {"summary.json", "payload.json"}
    mlflow.set_tracking_uri(None)


@pytest.mark.slow
@pytest.mark.parametrize("candidate", ["xgboost", "logreg"])
def test_mlflow_registers_each_model_type_in_its_native_format(candidate, reference, tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    sample = reference.head(600)
    matrix = model_matrix(sample)
    model = build_candidate(candidate).fit(matrix, sample[TARGET])

    tracker = Tracker(uri, "creditwatch-teste")
    with tracker.run("treino"):
        tracker.log_model(model, matrix.head(50), registered_name="credit-scoring-teste")

    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    versions = client.search_model_versions("name='credit-scoring-teste'")
    assert len(versions) == 1
    loaded = mlflow.pyfunc.load_model(f"models:/credit-scoring-teste/{versions[0].version}")
    assert len(loaded.predict(matrix.head(20))) == 20
    mlflow.set_tracking_uri(None)
