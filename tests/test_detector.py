from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditwatch.drift.detector import (
    CRITICAL,
    OK,
    WARNING,
    DriftThresholds,
    classify,
    detect_drift,
    detect_feature_drift,
)

THRESHOLDS = DriftThresholds()


@pytest.mark.parametrize(
    ("psi", "statistic", "p_value", "expected"),
    [
        (0.02, 0.03, 0.40, OK),
        (0.02, 0.03, 0.001, OK),  # significativo, mas efeito irrelevante: nao alerta
        (0.02, 0.15, 0.20, OK),  # efeito grande em amostra pequena, sem significancia
        (0.02, 0.15, 0.001, WARNING),  # o teste de hipotese eleva para atencao
        (0.12, 0.01, 0.90, WARNING),
        (0.25, 0.01, 0.90, CRITICAL),
        (0.90, 0.40, 0.0, CRITICAL),
    ],
)
def test_classify(psi, statistic, p_value, expected):
    assert classify(psi, statistic, p_value, THRESHOLDS) == expected


def _frame(rng: np.random.Generator, n: int, income_shift: float = 0.0, delay_rate: float = 0.2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "limit_bal": rng.lognormal(11 + income_shift, 0.8, n),
            "age": rng.integers(21, 70, n),
            "pay_status_1": (rng.random(n) < delay_rate).astype(int),
            "sex": rng.integers(1, 3, n),
        }
    )


def test_detect_drift_flags_only_the_features_that_changed():
    rng = np.random.default_rng(3)
    reference = _frame(rng, 12_000)
    current = _frame(rng, 3000, income_shift=0.8, delay_rate=0.6)
    report = detect_drift(reference, current)

    assert set(report.drifted_features) == {"limit_bal", "pay_status_1"}
    assert set(report.critical_features) == {"limit_bal", "pay_status_1"}
    assert report.share_drifted == pytest.approx(0.5)
    assert report.dataset_drift
    assert report.severity == CRITICAL


def test_detect_drift_is_silent_when_nothing_changed():
    rng = np.random.default_rng(4)
    report = detect_drift(_frame(rng, 12_000), _frame(rng, 3000))
    assert report.drifted_features == []
    assert not report.dataset_drift
    assert report.severity == OK


def test_detect_drift_ignores_columns_missing_from_one_side():
    rng = np.random.default_rng(5)
    report = detect_drift(_frame(rng, 2000), _frame(rng, 2000).drop(columns=["age"]))
    assert "age" not in {feature.feature for feature in report.features}


def test_feature_drift_uses_the_right_test_for_each_kind():
    rng = np.random.default_rng(6)
    reference, current = _frame(rng, 5000), _frame(rng, 2000)
    numerical = detect_feature_drift(reference["limit_bal"], current["limit_bal"], "numerical", THRESHOLDS)
    categorical = detect_feature_drift(reference["sex"], current["sex"], "categorical", THRESHOLDS)
    assert (numerical.test, categorical.test) == ("ks", "chi2")
    assert numerical.model_input
    assert not detect_feature_drift(reference["sex"], current["sex"], "categorical", THRESHOLDS, False).model_input


def test_report_serialization_sorts_features_by_psi():
    rng = np.random.default_rng(8)
    report = detect_drift(_frame(rng, 8000), _frame(rng, 3000, income_shift=0.8))
    payload = report.as_dict()
    psis = [feature["psi"] for feature in payload["features"]]
    assert psis == sorted(psis, reverse=True)
    assert payload["n_drifted"] == len(payload["drifted_features"])
    assert report.as_frame().iloc[0]["feature"] == "limit_bal"
    assert payload["features"][0]["drifted"] is True


def test_empty_report_has_no_drift():
    report = detect_drift(pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [1]}))
    assert report.features == []
    assert report.share_drifted == 0.0
    assert report.severity == OK
