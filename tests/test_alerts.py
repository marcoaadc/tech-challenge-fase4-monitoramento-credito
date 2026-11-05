from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditwatch.drift.detector import CRITICAL, WARNING, DriftThresholds, detect_drift
from creditwatch.observability.alerts import (
    CALIBRATION,
    DATA_DRIFT,
    DATA_QUALITY,
    LABEL_DEPENDENT,
    PERFORMANCE,
    PREDICTION_DRIFT,
    TARGET_DRIFT,
    AlertThresholds,
    add_months,
    data_quality_alert,
    drift_alerts,
    performance_alerts,
    prediction_drift_alert,
    read_alerts,
    write_alerts,
)

THRESHOLDS = AlertThresholds()
REFERENCE = {"roc_auc": 0.78, "default_rate": 0.22}


@pytest.mark.parametrize(
    ("month", "months", "expected"),
    [("2025-01", 0, "2025-01"), ("2025-01", 2, "2025-03"), ("2025-11", 2, "2026-01"), ("2025-12", 13, "2027-01")],
)
def test_add_months(month, months, expected):
    assert add_months(month, months) == expected


def _report(shift: dict[str, float], thresholds: DriftThresholds | None = None):
    rng = np.random.default_rng(11)
    columns = ["limit_bal", "age", "bill_amt_1", "bill_amt_2", "pay_amt_1", "pay_amt_2"]
    reference = pd.DataFrame({c: rng.normal(size=8000) for c in columns})
    current = pd.DataFrame({c: rng.normal(loc=shift.get(c, 0.0), size=3000) for c in columns})
    return detect_drift(reference, current, thresholds)


def test_no_drift_means_no_alert():
    assert drift_alerts("2025-01", _report({})) == []


def test_moderate_drift_raises_a_single_warning():
    alerts = drift_alerts("2025-04", _report({"age": 0.4}))
    assert len(alerts) == 1
    assert (alerts[0].kind, alerts[0].severity) == (DATA_DRIFT, WARNING)
    assert "age" in alerts[0].detail
    assert alerts[0].detectable_at == "2025-04"


def test_critical_feature_raises_a_single_critical_alert():
    alerts = drift_alerts("2025-05", _report({"age": 1.5}))
    assert len(alerts) == 1
    assert alerts[0].severity == CRITICAL
    assert alerts[0].metric == "max_feature_psi"
    assert alerts[0].value >= THRESHOLDS.psi_critical


def test_dataset_drift_is_consolidated_in_one_alert():
    report = _report({"age": 1.5, "limit_bal": 1.5, "bill_amt_1": 1.0})
    alerts = drift_alerts("2025-08", report)
    assert len(alerts) == 1  # um alerta por lote, nao um por feature
    assert alerts[0].title == "Drift do dataset"
    assert alerts[0].value == pytest.approx(0.5)
    assert alerts[0].as_dict()["threshold"] == report.thresholds.drift_share


@pytest.mark.parametrize(("psi", "severity"), [(0.05, None), (0.15, WARNING), (0.30, CRITICAL)])
def test_prediction_drift_alert(psi, severity):
    alert = prediction_drift_alert("2025-07", psi, THRESHOLDS)
    if severity is None:
        assert alert is None
    else:
        assert (alert.kind, alert.severity) == (PREDICTION_DRIFT, severity)


def _metrics(roc_auc: float, default_rate: float, mean_score: float) -> dict[str, float]:
    return {
        "roc_auc": roc_auc,
        "ks": 0.3,
        "default_rate": default_rate,
        "mean_score": mean_score,
        "calibration_gap": default_rate - mean_score,
    }


def test_healthy_model_raises_no_performance_alert():
    assert performance_alerts("2025-02", _metrics(0.775, 0.22, 0.22), REFERENCE, 0.8, THRESHOLDS, 2) == []


def test_auc_drop_alert_is_only_detectable_after_the_label_lag():
    alerts = performance_alerts("2025-11", _metrics(0.60, 0.22, 0.22), REFERENCE, 0.8, THRESHOLDS, 2)
    assert [alert.kind for alert in alerts] == [PERFORMANCE]
    assert alerts[0].severity == CRITICAL
    assert alerts[0].detectable_at == "2026-01"
    assert alerts[0].value == pytest.approx(0.18)


def test_small_auc_drop_is_a_warning():
    alerts = performance_alerts("2025-09", _metrics(0.745, 0.22, 0.22), REFERENCE, 0.8, THRESHOLDS, 2)
    assert [(alert.kind, alert.severity) for alert in alerts] == [(PERFORMANCE, WARNING)]


def test_calibration_and_target_drift_alerts():
    alerts = performance_alerts("2025-12", _metrics(0.78, 0.37, 0.30), REFERENCE, 1e-20, THRESHOLDS, 2)
    by_kind = {alert.kind: alert for alert in alerts}
    assert set(by_kind) == {CALIBRATION, TARGET_DRIFT}
    assert by_kind[CALIBRATION].severity == WARNING
    assert by_kind[TARGET_DRIFT].severity == CRITICAL
    assert set(by_kind) <= set(LABEL_DEPENDENT)


def test_target_drift_needs_statistical_significance():
    alerts = performance_alerts("2025-12", _metrics(0.78, 0.30, 0.30), REFERENCE, 0.40, THRESHOLDS, 2)
    assert alerts == []


def test_data_quality_alert_names_the_delivery_and_rules():
    alert = data_quality_alert("2025-07", "2025-07-r1", ["DQ-01", "DQ-09"], 2040)
    assert (alert.kind, alert.severity) == (DATA_QUALITY, CRITICAL)
    assert "2025-07-r1" in alert.detail
    assert "DQ-09" in alert.detail
    assert alert.value == 2.0


def test_alerts_roundtrip_as_json_lines(tmp_path):
    alerts = [
        data_quality_alert("2025-07", "2025-07-r1", ["DQ-01"], 10),
        prediction_drift_alert("2025-08", 0.3, THRESHOLDS),
    ]
    path = write_alerts(alerts, tmp_path / "nested" / "alerts.jsonl")
    assert read_alerts(path) == [alert.as_dict() for alert in alerts]
    assert read_alerts(write_alerts([], tmp_path / "empty.jsonl")) == []
    assert read_alerts(tmp_path / "missing.jsonl") == []
