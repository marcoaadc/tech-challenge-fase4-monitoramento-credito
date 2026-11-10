from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditwatch.drift.statistics import (
    chi_square_test,
    ks_test,
    psi_categorical,
    psi_numerical,
    quantile_edges,
    two_proportion_test,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(7)


def test_psi_is_zero_for_identical_samples(rng):
    sample = rng.normal(size=5000)
    assert psi_numerical(sample, sample) == pytest.approx(0.0, abs=1e-12)


def test_psi_is_small_for_two_samples_of_the_same_distribution(rng):
    assert psi_numerical(rng.normal(size=20_000), rng.normal(size=5000)) < 0.01


def test_psi_grows_with_the_size_of_the_shift(rng):
    reference = rng.normal(size=20_000)
    small = psi_numerical(reference, rng.normal(loc=0.2, size=5000))
    large = psi_numerical(reference, rng.normal(loc=1.0, size=5000))
    assert 0 < small < 0.10 < 0.25 < large


def test_psi_is_symmetric_in_sign_of_the_shift(rng):
    reference = rng.normal(size=50_000)
    up = psi_numerical(reference, rng.normal(loc=0.5, size=20_000))
    down = psi_numerical(reference, rng.normal(loc=-0.5, size=20_000))
    assert up == pytest.approx(down, rel=0.15)


def test_quantile_edges_merge_ties_in_zero_inflated_data():
    reference = np.concatenate([np.zeros(700), np.arange(1, 301)])
    edges = quantile_edges(reference)
    assert np.all(np.diff(edges) > 0)
    assert edges[0] == -np.inf
    assert edges[-1] == np.inf
    assert len(edges) < 11  # sete decis iguais a zero viram uma unica borda


def test_psi_handles_zero_inflated_data(rng):
    reference = np.where(rng.random(10_000) < 0.3, 0, rng.exponential(1000, 10_000))
    same = np.where(rng.random(5000) < 0.3, 0, rng.exponential(1000, 5000))
    more_zeros = np.where(rng.random(5000) < 0.7, 0, rng.exponential(1000, 5000))
    assert psi_numerical(reference, same) < 0.01
    assert psi_numerical(reference, more_zeros) > 0.25


def test_psi_categorical_detects_a_change_in_proportions():
    reference = pd.Series([1] * 500 + [2] * 500)
    assert psi_categorical(reference, reference) == pytest.approx(0.0, abs=1e-12)
    assert psi_categorical(reference, pd.Series([1] * 900 + [2] * 100)) > 0.25


def test_psi_categorical_handles_a_category_unseen_in_the_reference():
    reference = pd.Series([1] * 500 + [2] * 500)
    current = pd.Series([1] * 400 + [2] * 400 + [3] * 200)
    assert np.isfinite(psi_categorical(reference, current))
    assert psi_categorical(reference, current) > 0.25


def test_psi_rejects_an_empty_sample():
    with pytest.raises(ValueError, match="amostra vazia"):
        psi_categorical(pd.Series([1, 2]), pd.Series([], dtype="int64"))


def test_ks_test_separates_shifted_from_equal_distributions(rng):
    reference = rng.normal(size=5000)
    same = ks_test(reference, rng.normal(size=5000))
    shifted = ks_test(reference, rng.normal(loc=0.5, size=5000))
    assert same.p_value > 0.01
    assert same.statistic < 0.05
    assert shifted.p_value < 1e-10
    assert shifted.statistic > 0.15


def test_chi_square_reports_total_variation_distance():
    reference = pd.Series([0] * 800 + [1] * 200)
    same = chi_square_test(reference, reference)
    different = chi_square_test(reference, pd.Series([0] * 200 + [1] * 800))
    assert same.statistic == pytest.approx(0.0, abs=1e-9)
    assert same.p_value == pytest.approx(1.0)
    assert different.statistic == pytest.approx(0.6)  # 60% da carteira mudou de categoria
    assert different.p_value < 1e-10


@pytest.mark.parametrize(("n_reference", "n_current"), [(1000, 1000), (15_000, 2000), (15_000, 1000), (3000, 30_000)])
def test_categorical_effect_size_does_not_depend_on_sample_sizes(n_reference, n_current):
    """A mesma mudanca de proporcao (20% -> 35%) tem de dar o mesmo efeito com qualquer tamanho de lote."""
    reference = pd.Series([1] * int(0.20 * n_reference) + [0] * int(0.80 * n_reference))
    current = pd.Series([1] * int(0.35 * n_current) + [0] * int(0.65 * n_current))
    assert chi_square_test(reference, current).statistic == pytest.approx(0.15, abs=1e-9)


def test_chi_square_with_a_single_category_is_not_significant():
    constant = pd.Series([1] * 100)
    result = chi_square_test(constant, constant)
    assert result.statistic == 0.0
    assert result.p_value == 1.0


def test_two_proportion_test():
    equal = two_proportion_test(220, 1000, 440, 2000)
    assert equal.statistic == pytest.approx(0.0)
    assert equal.p_value == pytest.approx(1.0)
    higher = two_proportion_test(220, 1000, 700, 2000)
    assert higher.statistic == pytest.approx(0.13)
    assert higher.p_value < 1e-10


def test_two_proportion_test_without_variance():
    assert two_proportion_test(0, 100, 0, 100).p_value == 1.0
