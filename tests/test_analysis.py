from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditwatch.analysis.causal import (
    SEGMENTS,
    assign_segment,
    calibration_scan,
    root_cause_ranking,
    segment_diagnosis,
    shapley_values,
    utilization_cut,
)
from creditwatch.analysis.fairness import (
    ADVERSE,
    FOUR_FIFTHS,
    INCONCLUSIVE,
    WITHIN_LIMIT,
    age_band,
    fairness_audit,
    group_metrics,
    minimum_group_size,
    ratio_confidence_interval,
    with_group_labels,
)
from creditwatch.analysis.remediation import (
    STRATEGIES,
    average_folds,
    bad_rate_at_approval,
    recalibrate_by_segment,
    remediation_experiment,
    segment_calibration_factors,
)
from creditwatch.data.schema import BATCH_MONTH, DECISION, ID_COLUMN, SCORE, TARGET
from creditwatch.data.split import SPLIT_COLUMN
from creditwatch.features.engineering import model_matrix
from creditwatch.model.training import build_candidate
from creditwatch.simulation.scenario import SOURCE_ID, batch_lineage, default_scenario, generate_batch

CUT = 0.5


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


def test_assign_segment_uses_the_given_cut():
    frame = pd.DataFrame({"pay_status_1": [0, -1, 2, 1], "utilization_1": [0.1, 0.9, 0.9, 0.0]})
    assert assign_segment(frame, CUT).tolist() == [SEGMENTS[0], SEGMENTS[1], SEGMENTS[2], SEGMENTS[2]]
    assert assign_segment(frame, 0.05).tolist() == [SEGMENTS[1], SEGMENTS[1], SEGMENTS[2], SEGMENTS[2]]
    assert utilization_cut(frame) == pytest.approx(0.5)  # mediana dos dados, nao uma constante do simulador


def _portfolio() -> pd.DataFrame:
    """Carteira em que o modelo esta calibrado em todos os segmentos."""
    return pd.DataFrame(
        {
            "pay_status_1": [0] * 800 + [2] * 200,
            "utilization_1": [0.1] * 400 + [0.8] * 400 + [0.5] * 200,
            # Limite sorteado sem relacao com o segmento nem com o rotulo.
            "limit_bal": np.random.default_rng(3).permutation(np.tile(np.arange(1, 201) * 1000, 5)),
            SCORE: [0.1] * 800 + [0.5] * 200,
            TARGET: ([1] * 40 + [0] * 360) + ([1] * 40 + [0] * 360) + ([1] * 100 + [0] * 100),
        }
    )


def _portfolio_after_concept_drift() -> pd.DataFrame:
    drifted = _portfolio()
    drifted.loc[400:799, TARGET] = [1] * 200 + [0] * 200  # so quem esta em dia e usa muito limite piora
    return drifted


def test_segment_diagnosis_exposes_the_segment_where_the_relation_changed():
    rows = segment_diagnosis({"referencia": _portfolio(), "producao": _portfolio_after_concept_drift()}, CUT)
    gaps = {(row["period"], row["segment"]): row["gap"] for row in rows}
    assert gaps[("referencia", SEGMENTS[1])] == pytest.approx(0.0)
    assert gaps[("producao", SEGMENTS[1])] == pytest.approx(0.4)
    assert gaps[("producao", SEGMENTS[0])] == pytest.approx(0.0)
    assert sum(row["share"] for row in rows if row["period"] == "referencia") == pytest.approx(1.0)


def test_calibration_scan_finds_the_features_that_isolate_the_error_without_being_told():
    scan = calibration_scan(
        _portfolio(), _portfolio_after_concept_drift(), ("limit_bal", "utilization_1", "pay_status_1")
    )
    ranking = [row["feature"] for row in scan["by_feature"]]
    assert ranking[0] == "utilization_1"
    assert ranking[-1] == "limit_bal"  # nao tem relacao com o erro: excesso igual em todas as faixas
    top = scan["by_feature"][0]
    assert top["worst_bin"] == "0.8"
    assert top["worst_excess_gap"] == pytest.approx(0.4)
    assert top["spread"] == pytest.approx(0.4)
    assert scan["by_feature"][-1]["spread"] < top["spread"] / 2
    assert scan["bins"][0]["excess_gap"] >= scan["bins"][-1]["excess_gap"]


def test_calibration_scan_ignores_bins_too_small_to_estimate():
    reference, current = _portfolio(), _portfolio_after_concept_drift()
    current.loc[current.index[:20], "utilization_1"] = 0.3  # so 20 linhas nessa categoria
    scan = calibration_scan(reference, current, ("utilization_1",))
    assert "0.3" not in {row["bin"] for row in scan["bins"]}


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
    assert by_group["masculino"]["disparate_impact_ci"] == [1.0, 1.0]
    assert by_group["feminino"]["disparate_impact"] < FOUR_FIFTHS
    assert by_group["feminino"]["adverse_impact"]
    assert by_group["feminino"]["verdict"] == ADVERSE
    assert by_group["masculino"]["verdict"] == WITHIN_LIMIT
    assert by_group["feminino"]["equal_opportunity_gap"] > 0.2


def test_ratio_confidence_interval_narrows_with_sample_size():
    low_small, high_small = ratio_confidence_interval(0.6, 200, 0.7, 200)
    low_large, high_large = ratio_confidence_interval(0.6, 20_000, 0.7, 20_000)
    assert low_small < low_large < 0.6 / 0.7 < high_large < high_small
    assert low_small < FOUR_FIFTHS < low_large  # so a amostra grande sustenta "dentro do limite"
    assert np.isnan(ratio_confidence_interval(0.0, 100, 0.7, 100)[0])


def test_verdict_is_inconclusive_when_only_the_interval_crosses_the_limit():
    frame = _decisions(0.70, 0.60, n=150)
    # Taxas exatas: 105 de 150 aprovados em um grupo (70%) e 93 de 150 no outro (62%); razao 0,886.
    frame[DECISION] = [True] * 105 + [False] * 45 + [True] * 93 + [False] * 57
    by_group = {row["group"]: row for row in group_metrics(with_group_labels(frame), "sex")}
    worst = min(by_group.values(), key=lambda row: row["disparate_impact"])
    assert worst["disparate_impact"] >= FOUR_FIFTHS
    assert worst["disparate_impact_ci"][0] < FOUR_FIFTHS
    assert worst["verdict"] == INCONCLUSIVE
    assert not worst["adverse_impact"]


def test_small_groups_are_reported_but_not_compared():
    frame = with_group_labels(_decisions(0.8, 0.8))
    frame.loc[frame.index[:50], "marriage"] = "outros"
    rows = {row["group"]: row for row in group_metrics(frame, "marriage")}
    assert rows["casado(a)"]["compared"]
    assert not rows["outros"]["compared"]
    assert rows["outros"]["n"] == 50
    assert "disparate_impact" not in rows["outros"]


def test_small_group_cannot_become_the_reference_group():
    """Um grupo com 3% de uma amostra grande fica de fora: sua taxa de aprovacao e ruidosa demais."""
    assert minimum_group_size(2000) == 100
    assert minimum_group_size(10_000) == 200
    frame = with_group_labels(_decisions(0.70, 0.66, n=5000))
    frame.loc[frame.index[:150], "marriage"] = "outros"
    frame.loc[frame.index[:150], DECISION] = True  # grupo pequeno aprovado em 100%
    groups = {row["group"]: row for row in group_metrics(frame, "marriage")}
    assert groups["casado(a)"]["disparate_impact"] == pytest.approx(1.0)
    assert not groups["casado(a)"]["adverse_impact"]
    assert not groups["outros"]["compared"]


def test_fairness_audit_reports_the_least_approved_group_per_period():
    audit = fairness_audit({"referencia": _decisions(0.8, 0.78), "producao": _decisions(0.8, 0.5)})
    worst = {(row["period"], row["attribute"]): row for row in audit["worst_groups"]}
    assert not worst[("referencia", "sex")]["adverse_impact"]
    assert worst[("producao", "sex")]["adverse_impact"]
    assert worst[("producao", "sex")]["group"] == "feminino"
    assert worst[("producao", "sex")]["n"] == 1000
    assert audit["four_fifths_threshold"] == FOUR_FIFTHS
    assert set(audit["by_period"]) == {"referencia", "producao"}
    largest = audit["largest_equal_opportunity_gap"]
    assert (largest["period"], largest["group"]) == ("producao", "feminino")
    assert largest["equal_opportunity_gap"] > 0.2


def test_bad_rate_at_approval_uses_the_same_volume_for_every_strategy():
    y = np.array([0, 0, 1, 0, 1, 1, 0, 1, 1, 1])
    good_ranking = np.linspace(0.05, 0.95, 10)  # os quatro de menor score: tres bons e um mau
    assert bad_rate_at_approval(y, good_ranking, 0.4) == pytest.approx(0.25)
    assert bad_rate_at_approval(y, good_ranking[::-1].copy(), 0.4) == pytest.approx(0.75)
    assert bad_rate_at_approval(y, good_ranking, 1.0) == pytest.approx(0.6)
    assert np.isnan(bad_rate_at_approval(y, good_ranking, 0.0))


def test_segment_recalibration_corrects_only_the_segment_that_drifted():
    recent = pd.DataFrame(
        {
            "pay_status_1": [0] * 600 + [2] * 200,
            "utilization_1": [0.1] * 300 + [0.8] * 300 + [0.5] * 200,
            TARGET: ([1] * 30 + [0] * 270) + ([1] * 120 + [0] * 180) + ([1] * 100 + [0] * 100),
        }
    )
    score = np.array([0.10] * 600 + [0.50] * 200)
    factors = segment_calibration_factors(recent, score, CUT)
    assert factors[SEGMENTS[0]] == pytest.approx(1.0)
    assert factors[SEGMENTS[1]] == pytest.approx(4.0)  # observada 40% contra 10% prevista
    assert factors[SEGMENTS[2]] == pytest.approx(1.0)

    adjusted = recalibrate_by_segment(recent, score, factors, CUT)
    assert adjusted[:300] == pytest.approx(0.10)
    assert adjusted[300:600] == pytest.approx(0.40)
    assert adjusted[600:] == pytest.approx(0.50)
    assert recalibrate_by_segment(recent, np.full(800, 0.9), factors, CUT).max() == 1.0  # PD nunca passa de 1


def test_small_segments_are_not_recalibrated():
    recent = pd.DataFrame({"pay_status_1": [0] * 50, "utilization_1": [0.9] * 50, TARGET: [1] * 50})
    assert segment_calibration_factors(recent, np.full(50, 0.1), CUT) == {SEGMENTS[1]: 1.0}


@pytest.fixture(scope="module")
def backtest(reference, pool):
    """Champion e lotes rotulados gerados de duas metades disjuntas do pool."""
    scenario = default_scenario()
    half = len(pool) // 2
    train_pool, test_pool = pool.iloc[:half], pool.iloc[half:]

    def batches(source: pd.DataFrame, plans) -> pd.DataFrame:
        return pd.concat(
            [generate_batch(source, plan, 800).merge(batch_lineage(source, plan, 800), on=ID_COLUMN) for plan in plans],
            ignore_index=True,
        )

    train = reference[reference[SPLIT_COLUMN] == "train"].head(3000)
    champion = build_candidate("logreg").fit(model_matrix(train), train[TARGET])
    return {
        "champion": champion,
        "champion_threshold": 0.3,
        "model_type": "logreg",
        "reference_train": train,
        "train_production": batches(train_pool, scenario[6:9]),  # jul, ago, set
        "test_production": batches(test_pool, scenario[10:12]),  # nov, dez
        "recent_months": ["2025-08", "2025-09"],
        "segment_cut": CUT,
        "group_column": SOURCE_ID,
        "label_lag_months": 2,
    }


def test_remediation_backtest_compares_all_strategies_on_unseen_customers(backtest):
    result = remediation_experiment(**backtest)
    assert set(result["strategies"]) == set(STRATEGIES)
    assert result["disjoint_customers"]
    assert (result["train_months"], result["test_months"]) == (
        ["2025-07", "2025-08", "2025-09"],
        ["2025-11", "2025-12"],
    )
    assert result["recent_months"] == ["2025-08", "2025-09"]
    assert result["test_rows"] == 1600
    assert result["strategies"]["retreino_janela_recente"]["train_rows"] == 1600
    assert result["strategies"]["retreino_referencia_mais_producao"]["train_rows"] == 3000 + 2400
    champion = result["strategies"]["campeao"]
    # No volume de aprovacao do proprio campeao, as duas medidas de inadimplencia coincidem.
    assert champion["bad_rate_at_champion_approval"] == pytest.approx(champion["bad_rate_approved"], abs=2e-3)
    assert set(result["segment_calibration_factors"]) <= set(SEGMENTS)

    averaged = average_folds([result, result])
    assert averaged["folds"] == 2
    assert averaged["test_rows"] == 3200
    assert averaged["strategies"]["campeao"]["roc_auc"] == pytest.approx(champion["roc_auc"])
    assert averaged["strategies"]["campeao"]["roc_auc_by_fold"] == [champion["roc_auc"]] * 2


def test_remediation_rejects_labels_that_would_not_be_mature(backtest):
    """Com atraso de 2 meses, o rotulo de outubro so existe em dezembro: nao pode treinar para testar novembro."""
    scenario = default_scenario()
    late = generate_batch(_pool_like(backtest), scenario[9], 400)  # lote de outubro, de clientes fora do teste
    late = late.assign(**{SOURCE_ID: -1 - np.arange(len(late))})
    train_with_october = pd.concat([backtest["train_production"], late], ignore_index=True)
    with pytest.raises(ValueError, match="so amadurece em 2025-12"):
        remediation_experiment(**{**backtest, "train_production": train_with_october})
    # Sem atraso de rotulo, o mesmo treino e aceito.
    accepted = remediation_experiment(**{**backtest, "train_production": train_with_october, "label_lag_months": 0})
    assert accepted["train_months"][-1] == "2025-10"


def _pool_like(backtest: dict) -> pd.DataFrame:
    """Um pool minimo para gerar um lote extra: as proprias linhas de treino, sem as colunas de lote."""
    return backtest["reference_train"].drop(columns=[SPLIT_COLUMN]).reset_index(drop=True)


def test_remediation_rejects_customers_shared_between_train_and_test(backtest):
    leaked = pd.concat([backtest["train_production"], backtest["test_production"].head(50)], ignore_index=True)
    leaked[BATCH_MONTH] = leaked[BATCH_MONTH].where(leaked[BATCH_MONTH] < "2025-10", "2025-09")
    with pytest.raises(ValueError, match="50 clientes aparecem no treino e no teste"):
        remediation_experiment(**{**backtest, "train_production": leaked})


def test_remediation_rejects_overlapping_months_and_empty_window(backtest):
    with pytest.raises(ValueError, match="nao podem estar no treino"):
        remediation_experiment(**{**backtest, "test_production": backtest["train_production"], "group_column": None})
    with pytest.raises(ValueError, match="janela recente"):
        remediation_experiment(**{**backtest, "recent_months": ["2024-01"]})
