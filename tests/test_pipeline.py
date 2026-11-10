"""Testes de integracao: o pipeline completo roda uma vez (fixture ``workspace``) e os artefatos sao verificados."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from creditwatch.analysis.remediation import CEILING, STRATEGIES
from creditwatch.dashboard.build import build_dashboard
from creditwatch.data.schema import ID_COLUMN, TARGET
from creditwatch.data.split import SPLIT_COLUMN
from creditwatch.model.registry import CreditScorer
from creditwatch.observability.alerts import read_alerts
from creditwatch.observability.events import StageTracker, read_events
from creditwatch.pipelines import validate
from creditwatch.pipelines._cli import read_json, read_table
from creditwatch.pipelines.monitor import ReferenceBundle, checkpoint_months, process_delivery, run_monitor
from creditwatch.pipelines.run_all import main as run_all_main
from creditwatch.pipelines.simulate import MANIFEST_FILENAME
from creditwatch.simulation.corruption import DEMO_DEFECTS, expected_rules

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def timeline(workspace) -> pd.DataFrame:
    return pd.read_csv(workspace.settings.reports_dir / "monitoring" / "timeline.csv")


def test_ingestion_blocks_raw_data_and_accepts_the_cleaned_reference(workspace):
    quality = workspace.settings.reports_dir / "data_quality"
    report = read_json(quality / "ingestion_report.json")
    assert report["raw_validation"] == {"status": "bloqueado", "violated_rules": ["DQ-05"], "rows_with_failures": 399}
    assert report["clean_validation"]["status"] == "aprovado"
    assert report["cleaning"]["duplicates_removed"] == 35
    assert report["source"]["license"] == "CC BY 4.0"
    assert read_json(quality / "raw_validation.json")["status"] == "bloqueado"


def test_reference_and_production_never_overlap(workspace):
    reference = read_table(workspace.settings.reference_path)
    pool = read_table(workspace.settings.production_pool_path)
    assert not set(reference[ID_COLUMN]) & set(pool[ID_COLUMN])
    assert set(reference[SPLIT_COLUMN]) == {"train", "val", "test"}
    assert reference[TARGET].mean() == pytest.approx(pool[TARGET].mean(), abs=0.002)


def test_trained_model_is_versioned_and_does_not_use_protected_attributes(workspace):
    scorer = CreditScorer.load(workspace.settings.models_dir)
    metadata = scorer.metadata
    assert metadata.model_type in ("xgboost", "logreg")
    assert metadata.metrics["test"]["roc_auc"] > 0.75
    assert not {"sex", "age", "marriage", "education"} & set(metadata.features)
    report = read_json(workspace.settings.reports_dir / "model" / "training_metrics.json")
    assert {candidate["name"] for candidate in report["candidates"]} == {"logreg", "xgboost"}
    assert report["model_version"] == metadata.model_version


def test_production_batches_arrive_without_labels(workspace):
    settings = workspace.settings
    manifest = read_json(settings.production_dir / MANIFEST_FILENAME)
    assert len(manifest["deliveries"]) == 13
    good = [delivery for delivery in manifest["deliveries"] if delivery["labels_file"]]
    assert len(good) == 12
    for delivery in good[:3]:
        batch = read_table(settings.production_dir / delivery["batch_file"])
        labels = read_table(settings.production_dir / delivery["labels_file"])
        assert TARGET not in batch.columns
        assert list(labels.columns) == [ID_COLUMN, TARGET]
        assert sorted(batch[ID_COLUMN]) == sorted(labels[ID_COLUMN])
    corrupted = next(delivery for delivery in manifest["deliveries"] if not delivery["labels_file"])
    assert corrupted["delivery"] == "2025-07-r1"
    assert corrupted["expected_rules"] == expected_rules(DEMO_DEFECTS)


def test_corrupted_delivery_is_blocked_and_quarantined(workspace, timeline):
    settings = workspace.settings
    blocked = timeline[timeline["status"] == "blocked"]
    assert blocked["delivery"].tolist() == ["2025-07-r1"]
    assert blocked["rows_scored"].tolist() == [0]
    report = read_json(settings.reports_dir / "data_quality" / "blocked_2025-07-r1.json")
    assert [violation["rule_id"] for violation in report["violations"]] == expected_rules(DEMO_DEFECTS)
    assert (settings.quarantine_dir / "2025-07-r1" / "batch.csv").exists()
    assert not (settings.data_dir / "predictions" / "predictions_2025-07-r1.parquet").exists()
    # A remessa corrigida do mesmo mes e processada normalmente.
    assert timeline.loc[timeline["delivery"] == "2025-07-r2", "status"].item() == "scored"


def test_stable_months_raise_no_false_alarm(timeline):
    stable = timeline[timeline["phase"] == "estavel"]
    assert len(stable) == 3
    assert (stable["n_drifted"] == 0).all()
    assert (stable["alerts_critical"] + stable["alerts_warning"] == 0).all()
    assert (stable["score_psi"] < 0.05).all()


def test_data_drift_is_detected_before_the_model_degrades(workspace, timeline):
    drift = timeline[(timeline["phase"] == "drift_de_dados") & (timeline["status"] == "scored")]
    assert drift["n_drifted"].iloc[-1] >= 10
    assert drift["share_drifted"].iloc[-1] >= workspace.settings.drift_share_alert
    # Drift de dados puro: as features mudam, mas o poder de discriminacao se mantem.
    assert drift["roc_auc"].min() > workspace.overview["reference_metrics"]["roc_auc"] - 0.03
    assert workspace.overview["first_drift_alert"] == "2025-04"


def test_concept_drift_degrades_the_model(workspace, timeline):
    concept = timeline[timeline["phase"] == "drift_de_conceito"]
    reference_auc = workspace.overview["reference_metrics"]["roc_auc"]
    # A queda e progressiva: cada bimestre pior que o anterior (mes a mes ha ruido de amostra).
    assert concept["roc_auc"].iloc[2:].mean() < concept["roc_auc"].iloc[:2].mean() - 0.05
    assert concept["roc_auc"].iloc[-1] < reference_auc - 0.10
    assert concept["bad_rate_approved"].iloc[-1] > 2 * workspace.overview["reference_metrics"]["bad_rate_approved"]
    # O primeiro alerta de performance sai na fase de conceito, nunca antes.
    assert workspace.overview["first_performance_alert"] in ("2025-09", "2025-10")
    assert concept["labels_available_at"].iloc[0] == "2025-11"


def test_reference_performance_is_measured_on_the_test_split_only(workspace):
    """A particao de validacao escolheu o modelo e o corte: nao pode servir de base de performance."""
    metadata = read_json(workspace.settings.models_dir / "metadata.json")
    reference = workspace.overview["reference_metrics"]
    assert reference["n"] == metadata["reference"]["splits"]["test"]
    assert reference["roc_auc"] == pytest.approx(metadata["metrics"]["test"]["roc_auc"], abs=1e-4)


def test_drifted_features_are_the_ones_the_scenario_changed(workspace):
    last = read_json(workspace.settings.reports_dir / "monitoring" / "batches" / "2025-12.json")
    drifted = set(last["drift"]["drifted_features"])
    assert {"utilization_1", "pay_status_1", "limit_bal", "age"} <= drifted
    assert "sex" not in drifted
    features = pd.read_csv(workspace.settings.reports_dir / "monitoring" / "feature_drift.csv")
    assert features["month"].nunique() == 12
    assert {"psi", "statistic", "p_value", "severity", "test"} <= set(features.columns)


def test_alerts_cover_every_family(workspace):
    alerts = read_alerts(workspace.settings.alerts_path)
    kinds = {alert["kind"] for alert in alerts}
    # O alerta de calibracao depende do tamanho da amostra; os demais tem de aparecer sempre.
    assert kinds >= {"data_quality", "data_drift", "prediction_drift", "target_drift", "performance"}
    assert kinds <= {"data_quality", "data_drift", "prediction_drift", "target_drift", "performance", "calibration"}
    assert workspace.overview["alerts"]["total"] == len(alerts)
    performance = [alert for alert in alerts if alert["kind"] == "performance"]
    assert all(alert["detectable_at"] > alert["batch"] for alert in performance)  # rotulo so chega depois
    per_batch = pd.Series([alert["batch"] for alert in alerts if alert["kind"] == "data_drift"]).value_counts()
    assert per_batch.max() == 1


def test_pipeline_events_and_metrics_are_centralized(workspace):
    settings = workspace.settings
    events = read_events(settings.events_path)
    assert {event["pipeline"] for event in events} == {"ingest", "train", "simulate", "monitor", "analyze", "dashboard"}
    assert len({event["run_id"] for event in events}) == 1
    monitor = [event for event in events if event["pipeline"] == "monitor"]
    assert sum(event["status"] == "blocked" for event in monitor) == 1
    assert not any(event["status"] == "failed" for event in events)
    assert workspace.overview["pipeline"]["blocked"] == 1

    prom = (settings.observability_dir / "metrics.prom").read_text(encoding="utf-8")
    assert 'creditwatch_batches_total{status="scored"} 12.0' in prom
    assert 'creditwatch_batches_total{status="blocked"} 1.0' in prom
    assert 'creditwatch_last_batch_info{batch="2025-12"' in prom
    assert "creditwatch_model_roc_auc" in prom


def test_predictions_are_persisted_with_the_model_version(workspace):
    predictions = read_table(workspace.settings.data_dir / "predictions" / "predictions_2025-12.parquet")
    assert list(predictions.columns) == [ID_COLUMN, "score", "approved", "model_version"]
    assert len(predictions) == workspace.settings.batch_size
    assert predictions["score"].between(0, 1).all()


def test_causal_attribution_blames_concept_drift_for_the_auc_drop(workspace):
    causal = workspace.analysis["causal"]
    shapley = causal["shapley"]["roc_auc"]
    assert min(shapley, key=shapley.get) == "concept"
    assert shapley["concept"] < -0.08
    assert abs(shapley["new_profiles"]) < 0.05
    assert causal["sample_size"] == workspace.settings.batch_size  # mesmo tamanho das remessas
    assert len(causal["seeds"]) == 2
    assert causal["shapley_std"]["roc_auc"]["concept"] < 0.05
    # Eficiencia: as parcelas somam exatamente o efeito total.
    assert sum(shapley.values()) == pytest.approx(causal["total_effect"]["roc_auc"], abs=1e-9)
    # O drift de conceito nao mexe nas features: nao explica nenhuma feature com drift.
    assert causal["shapley"]["n_drifted"]["concept"] == pytest.approx(0.0, abs=1e-9)
    assert causal["shapley"]["n_drifted"]["inflation"] > 3


def test_segment_diagnosis_points_to_on_time_customers_with_high_utilization(workspace):
    segments = {(row["period"], row["segment"]): row for row in workspace.analysis["segments"]}
    assert abs(segments[("referencia", "em dia, utilização alta")]["gap"]) < 0.03
    assert segments[("drift_de_conceito", "em dia, utilização alta")]["gap"] > 0.10
    assert abs(segments[("drift_de_conceito", "em dia, utilização baixa")]["gap"]) < 0.05


def test_calibration_scan_finds_the_affected_customers_without_simulator_knowledge(workspace):
    diagnosis = workspace.analysis["diagnosis"]
    # O corte de utilizacao vem da referencia (mediana), nao da constante do simulador (0,5).
    assert 0.2 < diagnosis["utilization_cut"] < 0.45
    top = [row["feature"] for row in diagnosis["calibration_scan"]["by_feature"][:8]]
    assert top[0] == "pay_status_1"  # quem esta em dia e quem o modelo passou a errar
    assert {"utilization_1", "bill_amt_1"} & set(top)
    worst = diagnosis["calibration_scan"]["by_feature"][0]
    assert worst["worst_bin"] == "0"
    assert worst["worst_excess_gap"] > 0.10


def test_remediation_backtest_has_no_temporal_or_customer_leakage(workspace):
    remediation = workspace.analysis["remediation"]
    strategies = remediation["strategies"]
    assert remediation["test_months"] == ["2025-11", "2025-12"]
    # Com atraso de dois meses, em novembro so ha rotulo ate setembro.
    assert remediation["label_lag_months"] == 2
    assert remediation["train_months"][-1] == "2025-09"
    assert remediation["disjoint_customers"]
    assert remediation["folds"] == 2
    assert set(strategies) == {*STRATEGIES, CEILING}
    # Nenhuma estrategia, nem o teto com rotulos imediatos, devolve o desempenho da referencia.
    reference_auc = workspace.overview["reference_metrics"]["roc_auc"]
    assert all(row["roc_auc"] < reference_auc - 0.05 for row in strategies.values())
    assert strategies[CEILING]["roc_auc"] > strategies["campeao"]["roc_auc"]


def test_shorter_label_lag_lets_the_recalibration_see_the_affected_segment(workspace):
    main = workspace.analysis["remediation"]
    shorter = main["shorter_label_lag"]
    assert shorter["label_lag_months"] == 1
    assert shorter["train_months"][-1] == "2025-10"
    assert shorter["test_months"] == main["test_months"]
    segment = "em dia, utilização alta"
    # Um mes a mais de rotulos revela o segmento que passou a inadimplir.
    assert shorter["segment_calibration_factors"][segment] > main["segment_calibration_factors"][segment] + 0.15
    assert shorter["strategies"]["campeao"]["roc_auc"] == main["strategies"]["campeao"]["roc_auc"]


def test_fairness_audit_covers_all_periods_and_attributes(workspace):
    fairness = workspace.analysis["fairness"]
    assert set(fairness["by_period"]) == {"referencia", "estavel", "drift_de_dados", "drift_de_conceito"}
    assert len(fairness["worst_groups"]) == 16
    for row in fairness["worst_groups"]:
        low, high = row["disparate_impact_ci"]
        assert low <= row["disparate_impact"] <= high
        assert row["verdict"] in ("impacto adverso", "inconclusivo", "dentro do limite")
    assert fairness["largest_equal_opportunity_gap"]["equal_opportunity_gap"] > 0
    # Grupos pequenos aparecem no relatorio, marcados como fora da comparacao.
    excluded = [row for rows in fairness["by_period"].values() for row in rows if not row["compared"]]
    assert excluded
    assert all(row["n"] < 200 for row in excluded)
    # Os atributos protegidos nao acrescentam poder de discriminacao: exclui-los nao custa desempenho.
    ablation = fairness["protected_attribute_ablation"]
    assert abs(ablation["gain_test"]) < 0.01
    assert ablation["without_protected"]["roc_auc_test"] > 0.76
    report = (workspace.settings.reports_dir / "analysis" / "analysis_report.md").read_text(encoding="utf-8")
    assert "Auditoria de viés" in report
    assert "Drift de conceito" in report


def test_dashboard_is_self_contained_and_carries_valid_data(workspace):
    html = workspace.dashboard.read_text(encoding="utf-8")
    assert "<script src=" not in html
    assert "https://" not in html
    assert "http://www.w3.org/2000/svg" in html
    start = html.index('<script type="application/json" id="payload">') + len(
        '<script type="application/json" id="payload">'
    )
    payload = json.loads(html[start : html.index("</script>", start)])
    assert payload["months"] == [f"2025-{m:02d}" for m in range(1, 13)]
    assert [phase["phase"] for phase in payload["phases"]] == ["estavel", "drift_de_dados", "drift_de_conceito"]
    assert len(payload["deliveries"]) == 13
    assert payload["analysis"]["causal"]["mechanisms"] == ["inflation", "stress", "new_profiles", "concept"]
    assert payload["analysis"]["scan"][0]["feature"] == "pay_status_1"
    assert "teto_rotulos_imediatos" in payload["analysis"]["remediation"]["strategy_labels"]
    assert len(payload["timeline"]["roc_auc"]) == 12
    assert {stage["stage"] for stage in payload["stage_stats"]} >= {"contract", "score", "drift", "performance"}
    assert len(payload["rules"]) >= 13


def test_checkpoint_months_are_the_last_month_of_each_phase(workspace):
    manifest = read_json(workspace.settings.production_dir / MANIFEST_FILENAME)
    assert checkpoint_months(manifest["deliveries"]) == {"2025-03", "2025-08", "2025-12"}
    with pytest.raises(ValueError, match="evidently deve ser"):
        run_monitor(workspace.settings, evidently="sempre")


def test_validate_cli_blocks_the_corrupted_file_with_exit_code_1(workspace, monkeypatch, capsys):
    settings = workspace.settings
    monkeypatch.setattr(validate, "get_settings", lambda: settings)
    corrupted = settings.production_dir / "batch_2025-07_remessa1_corrompida.parquet"
    with pytest.raises(SystemExit) as excinfo:
        validate.main([str(corrupted), "--no-quarantine"])
    assert excinfo.value.code == 1
    output = capsys.readouterr().out
    assert "BLOQUEADO" in output
    assert "DQ-02" in output
    assert "cpf" in output
    report = read_json(settings.reports_dir / "data_quality" / f"validation_{corrupted.stem}.json")
    assert report["status"] == "bloqueado"


def test_validate_cli_accepts_a_good_file(workspace, monkeypatch, capsys):
    monkeypatch.setattr(validate, "get_settings", lambda: workspace.settings)
    validate.main([str(workspace.settings.production_dir / "batch_2025-01.parquet")])
    assert "APROVADO" in capsys.readouterr().out


def test_run_all_cli_reuses_the_stage_functions(workspace, monkeypatch):
    calls: list[str] = []
    from creditwatch.pipelines import run_all

    monkeypatch.setattr(run_all, "get_settings", lambda: workspace.settings)
    for name in ("run_ingest", "run_train", "run_simulate", "run_analyze"):
        monkeypatch.setattr(run_all, name, lambda *args, _name=name, **kwargs: calls.append(_name))
    monkeypatch.setattr(run_all, "run_monitor", lambda *args, **kwargs: calls.append("run_monitor") or {})
    monkeypatch.setattr(run_all, "run_dashboard", lambda *args, **kwargs: calls.append("run_dashboard") or "x")
    events_before = workspace.settings.events_path.read_text(encoding="utf-8")
    try:
        run_all_main(["--evidently", "none"])
    finally:
        workspace.settings.events_path.write_text(events_before, encoding="utf-8")
    assert calls == ["run_ingest", "run_train", "run_simulate", "run_monitor", "run_analyze", "run_dashboard"]


def test_rejected_label_file_does_not_stop_the_monitoring(workspace):
    """Rotulos invalidos deixam a performance da remessa indisponivel, mas a remessa e escorada e o run segue."""
    settings = workspace.settings
    manifest = read_json(settings.production_dir / MANIFEST_FILENAME)
    delivery = dict(next(d for d in manifest["deliveries"] if d["delivery"] == "2025-02"))
    labels = read_table(settings.production_dir / delivery["labels_file"])
    scorer = CreditScorer.load(settings.models_dir)
    reference = ReferenceBundle.build(read_table(settings.reference_path), scorer)

    for name, broken, problem in (
        ("labels_invalid.parquet", labels.assign(**{TARGET: 7}), "DQ-12"),
        ("labels_incomplete.parquet", labels.head(len(labels) - 25), "faltam rotulos de 25"),
    ):
        broken.to_parquet(settings.production_dir / name, index=False)
        tracker = StageTracker("monitor")
        summary = process_delivery(
            {**delivery, "labels_file": name}, settings, scorer, reference, tracker, with_evidently=False
        )
        (settings.production_dir / name).unlink()

        assert summary["status"] == "scored"
        assert summary["rows_scored"] == settings.batch_size
        assert summary["performance"] is None
        assert problem in summary["labels"]["problem"]
        assert summary["drift"]["n_drifted"] == 0  # o drift de entrada continua sendo calculado
        assert [alert["title"] for alert in summary["alerts"]] == ["Arquivo de rótulos rejeitado"]
        statuses = {event.stage: event.status for event in tracker.events}
        assert statuses["performance"] == "blocked"
        assert statuses["score"] == "success"


def test_dashboard_renders_when_no_label_has_matured(workspace, tmp_path):
    """Sem rotulos, as colunas de performance nem existem: o dashboard tem de abrir e nao pode acusar degradacao."""
    import shutil

    reports = tmp_path / "reports"
    shutil.copytree(workspace.settings.reports_dir, reports)
    timeline_path = reports / "monitoring" / "timeline.csv"
    timeline = pd.read_csv(timeline_path)
    performance = ["roc_auc", "ks", "gini", "brier", "precision", "recall", "default_rate", "calibration_gap"]
    timeline.drop(columns=[*performance, "bad_rate_approved", "roc_auc_drop", "labels_available_at"]).to_csv(
        timeline_path, index=False
    )
    settings = workspace.settings.model_copy(update={"reports_dir": reports})

    html = build_dashboard(settings, tmp_path / "dashboard.html").read_text(encoding="utf-8")
    start = html.index('<script type="application/json" id="payload">') + len(
        '<script type="application/json" id="payload">'
    )
    payload = json.loads(html[start : html.index("</script>", start)])
    assert payload["timeline"]["roc_auc"] == [None] * 12
    assert payload["timeline"]["n_drifted"][-1] > 0
