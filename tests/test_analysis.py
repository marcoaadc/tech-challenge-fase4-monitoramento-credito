from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from creditwatch.analysis.causal import SEGMENTS, assign_segment, root_cause_ranking, segment_diagnosis, shapley_values
from creditwatch.analysis.fairness import (
    FOUR_FIFTHS,
    age_band,
    fairness_audit,
    group_metrics,
    minimum_group_size,
    with_group_labels,
)
from creditwatch.analysis.remediation import remediation_experiment
from creditwatch.data.schema import BATCH_MONTH, DECISION, SCORE, TARGET


def test_shapley_values_of_an_additive_game_recover_each_contribution():
    players = ("a", "b", "c")
    contribution = {"a": 1.0, "b": -2.0, "c": 0.5}
    values = {
        frozenset(subset): sum(contribution[p] for p in subset)
        for subset in [(), ("a",), ("b",), ("c",), ("a", "b"), ("a", "c"), ("b", "c"), ("a", "b", "c")]
    }
    assert shapley_values(values, players) == pytest.approx(contribution)


def test_shapley_values_split_an_interaction_equally_and_sum_to_the_total():
    values = {frozenset(): 0.0, frozenset({"a"}): 0.0, frozenset({"b"}): 0.0, frozenset({"a", "b"}): 4.0}
    attribution = shapley_values(values, ("a", "b"))
    assert attribution == pytest.approx({"a": 2.0, "b": 2.0})
    assert sum(attribution.values()) == pytest.approx(values[frozenset({"a", "b"})] - values[frozenset()])


def test_assign_segment():
    frame = pd.DataFrame({"pay_status_1": [0, -1, 2, 1], "utilization_1": [0.1, 0.9, 0.9, 0.0]})
    assert assign_segment(frame).tolist() == [SEGMENTS[0], SEGMENTS[1], SEGMENTS[2], SEGMENTS[2]]


def test_segment_diagnosis_exposes_the_segment_where_the_relation_changed():
    base = pd.DataFrame(
        {
            "pay_status_1": [0] * 400 + [2] * 100,
            "utilization_1": [0.1] * 200 + [0.8] * 200 + [0.5] * 100,
            SCORE: [0.1] * 400 + [0.5] * 100,
            TARGET: ([1] * 20 + [0] * 180) + ([1] * 20 + [0] * 180) + ([1] * 50 + [0] * 50),
        }
    )
    drifted = base.copy()
    drifted.loc[200:399, TARGET] = [1] * 100 + [0] * 100  # so os alavancados passam a inadimplir
    rows = segment_diagnosis({"referencia": base, "producao": drifted})

    gaps = {(row["period"], row["segment"]): row["gap"] for row in rows}
    assert gaps[("referencia", SEGMENTS[1])] == pytest.approx(0.0)
    assert gaps[("producao", SEGMENTS[1])] == pytest.approx(0.4)
    assert gaps[("producao", SEGMENTS[0])] == pytest.approx(0.0)
    assert sum(row["share"] for row in rows if row["period"] == "referencia") == pytest.approx(1.0)


def test_root_cause_ranking_weights_drift_by_importance():
    drift = [
        {"feature": "utilization_1", "psi": 0.9, "severity": "critical"},
        {"feature": "pay_status_1", "psi": 0.3, "severity": "critical"},
        {"feature": "age", "psi": 0.5, "severity": "critical"},  # fora do modelo: nao entra
    ]
    ranking = root_cause_ranking(drift, {"utilization_1": 0.02, "pay_status_1": 0.20})
    assert [row["feature"] for row in ranking] == ["pay_status_1", "utilization_1"]
    assert ranking[0]["impact"] == pytest.approx(0.06)


def test_age_band_and_group_labels():
    assert age_band(pd.Series([21, 29, 30, 45, 50, 79])).tolist() == ["18-29", "18-29", "30-39", "40-49", "50+", "50+"]
    frame = pd.DataFrame({"sex": [1, 2], "education": [1, 3], "marriage": [1, 2], "age": [25, 55]})
    labeled = with_group_labels(frame)
    assert labeled["sex"].tolist() == ["masculino", "feminino"]
    assert labeled["education"].tolist() == ["pós-graduação", "ensino médio"]
    assert labeled["age_band"].tolist() == ["18-29", "50+"]


def _decisions(approval_a: float, approval_b: float, n: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    frame = pd.DataFrame(
        {
            "sex": [1] * n + [2] * n,
            "education": 1,
            "marriage": 1,
            "age": 35,
            TARGET: (rng.random(2 * n) < 0.2).astype(int),
            SCORE: rng.random(2 * n),
        }
    )
    frame[DECISION] = np.concatenate([rng.random(n) < approval_a, rng.random(n) < approval_b])
    return frame


def test_group_metrics_flag_adverse_impact_below_four_fifths():
    rows = group_metrics(with_group_labels(_decisions(0.8, 0.5)), "sex")
    by_group = {row["group"]: row for row in rows}
    assert by_group["masculino"]["disparate_impact"] == pytest.approx(1.0)
    assert by_group["feminino"]["disparate_impact"] < FOUR_FIFTHS
    assert by_group["feminino"]["adverse_impact"]
    assert not by_group["masculino"]["adverse_impact"]
    assert by_group["feminino"]["equal_opportunity_gap"] > 0.2


def test_group_metrics_skip_small_groups():
    frame = with_group_labels(_decisions(0.8, 0.8))
    frame.loc[frame.index[:50], "marriage"] = "outros"
    assert {row["group"] for row in group_metrics(frame, "marriage")} == {"casado(a)"}


def test_small_group_cannot_become_the_reference_group():
    """Um grupo com 3% de uma amostra grande fica de fora: sua taxa de aprovacao e ruidosa demais."""
    assert minimum_group_size(2000) == 100
    assert minimum_group_size(10_000) == 200
    frame = with_group_labels(_decisions(0.70, 0.66, n=5000))
    frame.loc[frame.index[:150], "marriage"] = "outros"
    frame.loc[frame.index[:150], DECISION] = True  # grupo pequeno aprovado em 100%
    groups = {row["group"]: row for row in group_metrics(frame, "marriage")}
    assert set(groups) == {"casado(a)"}
    assert not groups["casado(a)"]["adverse_impact"]


def test_fairness_audit_reports_the_least_approved_group_per_period():
    audit = fairness_audit({"referencia": _decisions(0.8, 0.78), "producao": _decisions(0.8, 0.5)})
    worst = {(row["period"], row["attribute"]): row for row in audit["worst_groups"]}
    assert not worst[("referencia", "sex")]["adverse_impact"]
    assert worst[("producao", "sex")]["adverse_impact"]
    assert worst[("producao", "sex")]["group"] == "feminino"
    assert audit["four_fifths_threshold"] == FOUR_FIFTHS
    assert set(audit["by_period"]) == {"referencia", "producao"}


def test_remediation_rejects_test_months_seen_in_training(reference):
    champion = DummyClassifier(strategy="prior").fit([[0]], [0])
    production = reference.head(100).assign(**{BATCH_MONTH: "2025-01"})
    with pytest.raises(ValueError, match="nao podem estar no treino"):
        remediation_experiment(champion, 0.5, "logreg", reference, production, ["2025-01"], ["2025-01"], ["2025-01"])
    with pytest.raises(ValueError, match="faltam meses"):
        remediation_experiment(champion, 0.5, "logreg", reference, production, ["2025-01"], ["2025-01"], ["2025-02"])
