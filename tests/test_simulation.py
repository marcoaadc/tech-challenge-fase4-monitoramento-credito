from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditwatch.data.schema import BATCH_MONTH, BILL_AMT, ID_COLUMN, LIMIT, PAY_AMT, PAY_STATUS, TARGET
from creditwatch.simulation.mechanisms import (
    LEVERAGED_UTILIZATION,
    PAYMENT_PASS_THROUGH,
    apply_concept_drift,
    apply_inflation,
    sample_batch,
    sample_positions,
    sampling_weights,
)
from creditwatch.simulation.scenario import (
    CONCEPT_DRIFT,
    DATA_DRIFT,
    MECHANISMS,
    SOURCE_ID,
    STABLE,
    MonthPlan,
    batch_ids,
    batch_lineage,
    default_scenario,
    generate_batch,
    month_seed,
)

FEATURES = [LIMIT, "sex", "education", "marriage", "age", *PAY_STATUS, *BILL_AMT, *PAY_AMT]


def test_sampling_is_uniform_without_mechanisms(pool):
    weights = sampling_weights(pool)
    assert weights.sum() == pytest.approx(1.0)
    assert np.allclose(weights, 1 / len(pool))


def test_new_profiles_brings_younger_customers_with_lower_limits(pool):
    rng = np.random.default_rng(0)
    batch = sample_batch(pool, 3000, rng, new_profiles=0.75)
    assert batch["age"].mean() < pool["age"].mean() - 1.5
    assert batch[LIMIT].median() < pool[LIMIT].median()


def test_stress_brings_more_delinquent_customers_and_more_defaults(pool):
    rng = np.random.default_rng(0)
    batch = sample_batch(pool, 3000, rng, stress=0.5)
    assert (batch[PAY_STATUS[0]] >= 1).mean() > (pool[PAY_STATUS[0]] >= 1).mean() + 0.05
    assert batch[TARGET].mean() > pool[TARGET].mean() + 0.03  # P(y|x) preservada: o rotulo acompanha


def test_sample_batch_draws_without_replacement(pool):
    batch = sample_batch(pool, 2000, np.random.default_rng(1), new_profiles=0.75, stress=0.5)
    assert batch[ID_COLUMN].is_unique
    assert len(batch) == 2000


def test_inflation_scales_bills_fully_and_payments_partially(stable_batch):
    inflated = apply_inflation(stable_batch, 0.60)
    positive = stable_batch[BILL_AMT[0]] > 1000
    ratio = (inflated.loc[positive, BILL_AMT[0]] / stable_batch.loc[positive, BILL_AMT[0]]).median()
    assert ratio == pytest.approx(1.60, abs=0.01)
    oldest = stable_batch[BILL_AMT[-1]] > 1000
    oldest_ratio = (inflated.loc[oldest, BILL_AMT[-1]] / stable_batch.loc[oldest, BILL_AMT[-1]]).median()
    assert oldest_ratio == pytest.approx(1.10, abs=0.01)
    paid = stable_batch[PAY_AMT[0]] > 1000
    pay_ratio = (inflated.loc[paid, PAY_AMT[0]] / stable_batch.loc[paid, PAY_AMT[0]]).median()
    assert pay_ratio == pytest.approx(1 + PAYMENT_PASS_THROUGH * 0.60, abs=0.01)


def test_inflation_keeps_limit_label_and_integer_types(stable_batch):
    inflated = apply_inflation(stable_batch, 0.5)
    pd.testing.assert_series_equal(inflated[LIMIT], stable_batch[LIMIT])
    pd.testing.assert_series_equal(inflated[TARGET], stable_batch[TARGET])
    assert str(inflated[BILL_AMT[0]].dtype) == "int64"
    assert apply_inflation(stable_batch, 0.0) is stable_batch


def test_concept_drift_changes_labels_but_not_features(stable_batch):
    drifted = apply_concept_drift(stable_batch, 1.0, np.random.default_rng(2))
    pd.testing.assert_frame_equal(drifted[FEATURES], stable_batch[FEATURES])
    with pytest.raises(ValueError, match="informe rng ou draws"):
        apply_concept_drift(stable_batch, 1.0)
    assert (drifted[TARGET] != stable_batch[TARGET]).sum() > 100
    assert apply_concept_drift(stable_batch, 0.0, np.random.default_rng(2)) is stable_batch


def test_concept_drift_only_touches_the_documented_segments(stable_batch):
    drifted = apply_concept_drift(stable_batch, 1.0, np.random.default_rng(2))
    changed = drifted[TARGET] != stable_batch[TARGET]
    on_time = stable_batch[PAY_STATUS[0]] <= 0
    leveraged = stable_batch[BILL_AMT[0]] / stable_batch[LIMIT] >= LEVERAGED_UTILIZATION

    new_defaults = changed & (drifted[TARGET] == 1)
    cured = changed & (drifted[TARGET] == 0)
    assert new_defaults.any()
    assert cured.any()
    assert (on_time & leveraged)[new_defaults].all()
    assert (~on_time)[cured].all()
    # Em dia e pouco alavancado: ninguem muda de rotulo.
    assert not changed[on_time & ~leveraged].any()


def test_concept_drift_intensity_is_monotonic(stable_batch):
    def flips(intensity: float) -> int:
        drifted = apply_concept_drift(stable_batch, intensity, np.random.default_rng(2))
        return int((drifted[TARGET] != stable_batch[TARGET]).sum())

    assert flips(0.25) < flips(0.5) < flips(1.0)


def test_default_scenario_has_three_phases_in_order():
    scenario = default_scenario()
    assert [plan.month for plan in scenario] == [f"2025-{m:02d}" for m in range(1, 13)]
    assert [plan.phase for plan in scenario] == [STABLE] * 3 + [DATA_DRIFT] * 5 + [CONCEPT_DRIFT] * 4
    for plan in scenario:
        intensities = [getattr(plan, name) for name in MECHANISMS]
        if plan.phase == STABLE:
            assert intensities == [0.0] * 4
        if plan.phase == DATA_DRIFT:
            assert plan.concept == 0.0
            assert plan.inflation > 0
        if plan.phase == CONCEPT_DRIFT:
            assert plan.concept > 0
    assert default_scenario(2030)[0].month == "2030-01"


def test_scenario_intensities_never_decrease():
    scenario = default_scenario()
    for name in MECHANISMS:
        values = [getattr(plan, name) for plan in scenario]
        assert values == sorted(values)


def test_month_plan_only_isolates_mechanisms():
    plan = default_scenario()[-1]
    isolated = plan.only("concept")
    assert (isolated.inflation, isolated.stress, isolated.new_profiles) == (0.0, 0.0, 0.0)
    assert isolated.concept == plan.concept
    assert plan.only().as_dict() == {**MonthPlan(plan.month, plan.phase).as_dict()}
    with pytest.raises(ValueError, match="mecanismos desconhecidos"):
        plan.only("inexistente")


def test_generate_batch_is_reproducible_and_month_specific(pool):
    plan = default_scenario()[8]
    first = generate_batch(pool, plan, 500, seed=42)
    second = generate_batch(pool, plan, 500, seed=42)
    other_seed = generate_batch(pool, plan, 500, seed=43)
    pd.testing.assert_frame_equal(first, second)
    assert not first[LIMIT].equals(other_seed[LIMIT])
    assert month_seed(42, "2025-01") != month_seed(42, "2025-02")


def test_generate_batch_assigns_month_and_unique_ids(pool):
    batch = generate_batch(pool, default_scenario()[3], 500)
    assert (batch[BATCH_MONTH] == "2025-04").all()
    assert batch[ID_COLUMN].is_unique
    assert batch[ID_COLUMN].min() == 202504 * 100_000 + 1
    assert batch_ids("2025-04", 3).tolist() == [20250400001, 20250400002, 20250400003]
    assert not set(batch[ID_COLUMN]) & set(pool[ID_COLUMN])


def test_batch_lineage_recovers_the_pool_customer_of_each_proposal(pool):
    plan = default_scenario()[9]
    batch = generate_batch(pool, plan, 800)
    lineage = batch_lineage(pool, plan, 800)
    assert lineage[ID_COLUMN].tolist() == batch[ID_COLUMN].tolist()
    assert lineage[SOURCE_ID].is_unique  # dentro do mes, a amostragem e sem reposicao
    assert set(lineage[SOURCE_ID]) <= set(pool[ID_COLUMN])
    # Idade e limite nao sao alterados por nenhum mecanismo: tem de coincidir com o cliente de origem.
    source = pool.set_index(ID_COLUMN).loc[lineage[SOURCE_ID]]
    assert (source["age"].to_numpy() == batch["age"].to_numpy()).all()
    assert (source[LIMIT].to_numpy() == batch[LIMIT].to_numpy()).all()
    assert SOURCE_ID not in batch.columns  # o lote de producao nao carrega a linhagem


def test_the_same_pool_customer_can_be_drawn_in_different_months(pool):
    scenario = default_scenario()
    first = batch_lineage(pool, scenario[0], 2000)
    second = batch_lineage(pool, scenario[1], 2000)
    assert set(first[SOURCE_ID]) & set(second[SOURCE_ID])


def test_sample_positions_follow_the_weights_and_reject_oversized_batches(pool):
    positions = sample_positions(pool, 3000, np.random.default_rng(0), stress=0.5)
    assert len(set(positions)) == 3000
    with pytest.raises(ValueError, match="maior que o pool"):
        sample_positions(pool, len(pool) + 1, np.random.default_rng(0))


def test_scenarios_with_the_same_seed_share_their_random_noise(pool):
    """Numeros aleatorios comuns: ligar um mecanismo nao pode embaralhar o ruido dos outros."""
    plan = default_scenario()[-1]
    full = batch_lineage(pool, plan, 2000)[SOURCE_ID]
    without_concept = batch_lineage(pool, plan.only("inflation", "stress", "new_profiles"), 2000)[SOURCE_ID]
    without_inflation = batch_lineage(pool, plan.only("stress", "new_profiles", "concept"), 2000)[SOURCE_ID]
    # Conceito e inflacao nao mexem na composicao da carteira: os mesmos clientes, na mesma ordem.
    assert full.tolist() == without_concept.tolist() == without_inflation.tolist()

    # Mudar a composicao troca apenas parte dos clientes; com ruido independente a sobreposicao seria de ~13%.
    baseline = batch_lineage(pool, plan.only(), 2000)[SOURCE_ID]
    only_stress = batch_lineage(pool, plan.only("stress"), 2000)[SOURCE_ID]
    assert len(set(baseline) & set(only_stress)) > 1200


def test_concept_drift_gives_each_customer_the_same_draw_in_every_scenario(pool):
    plan = default_scenario()[-1]
    with_inflation = generate_batch(pool, plan, 2000)
    without_inflation = generate_batch(pool, plan.only("stress", "new_profiles", "concept"), 2000)
    # A inflacao muda faturas, entao a utilizacao muda e alguns clientes entram ou saem do segmento
    # afetado; mas quem esta em atraso (curas) recebe exatamente o mesmo sorteio nos dois cenarios.
    delayed = with_inflation[PAY_STATUS[0]] >= 1
    assert with_inflation.loc[delayed, TARGET].tolist() == without_inflation.loc[delayed, TARGET].tolist()
