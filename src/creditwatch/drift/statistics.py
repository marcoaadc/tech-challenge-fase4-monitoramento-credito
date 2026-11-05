"""Testes estatisticos de drift implementados de forma transparente (numpy/scipy).

Servem de base auditavel para os alertas e de contraprova do relatorio do Evidently:

- **PSI** (Population Stability Index): tamanho do efeito, padrao da industria de credito.
  PSI < 0,10 estavel; 0,10-0,25 mudanca moderada; >= 0,25 mudanca significativa.
- **Kolmogorov-Smirnov** (numericas): maior distancia entre as distribuicoes acumuladas (D) e p-valor.
- **Qui-quadrado** (categoricas): independencia entre periodo e categoria, com V de Cramer.

Com milhares de linhas por lote o p-valor fica significativo para diferencas irrelevantes; por isso
as decisoes usam tamanho de efeito (PSI, D, V) e o p-valor entra apenas como evidencia.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

EPSILON = 1e-4


@dataclass(frozen=True)
class StatTest:
    statistic: float
    p_value: float


def _proportions(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    if total == 0:
        raise ValueError("amostra vazia")
    return np.clip(counts / total, EPSILON, None)


def _psi_from_counts(reference_counts: np.ndarray, current_counts: np.ndarray) -> float:
    expected = _proportions(reference_counts.astype(float))
    actual = _proportions(current_counts.astype(float))
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def quantile_edges(reference: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Bordas dos decis da referencia (bordas repetidas sao fundidas; extremos abertos)."""
    quantiles = np.quantile(reference, np.linspace(0, 1, n_bins + 1)[1:-1])
    inner = np.unique(quantiles)
    return np.concatenate(([-np.inf], inner, [np.inf]))


def psi_numerical(reference: pd.Series | np.ndarray, current: pd.Series | np.ndarray, n_bins: int = 10) -> float:
    """PSI com faixas definidas pelos quantis da referencia."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    edges = quantile_edges(ref, n_bins)
    # side="left": valores iguais a uma borda ficam na faixa inferior (estavel para dados com massa em zero).
    ref_counts = np.bincount(np.searchsorted(edges[1:-1], ref, side="left"), minlength=len(edges) - 1)
    cur_counts = np.bincount(np.searchsorted(edges[1:-1], cur, side="left"), minlength=len(edges) - 1)
    return _psi_from_counts(ref_counts, cur_counts)


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    """PSI sobre a uniao das categorias observadas nos dois periodos."""
    categories = sorted(set(pd.unique(reference)) | set(pd.unique(current)))
    ref_counts = pd.Series(reference).value_counts().reindex(categories, fill_value=0).to_numpy()
    cur_counts = pd.Series(current).value_counts().reindex(categories, fill_value=0).to_numpy()
    return _psi_from_counts(ref_counts, cur_counts)


def ks_test(reference: pd.Series | np.ndarray, current: pd.Series | np.ndarray) -> StatTest:
    """Teste de Kolmogorov-Smirnov para duas amostras."""
    result = stats.ks_2samp(np.asarray(reference, dtype=float), np.asarray(current, dtype=float))
    return StatTest(statistic=float(result.statistic), p_value=float(result.pvalue))


def chi_square_test(reference: pd.Series, current: pd.Series) -> StatTest:
    """Qui-quadrado de homogeneidade; ``statistic`` e o V de Cramer (tamanho do efeito em [0, 1])."""
    categories = sorted(set(pd.unique(reference)) | set(pd.unique(current)))
    table = np.vstack(
        [
            pd.Series(reference).value_counts().reindex(categories, fill_value=0).to_numpy(),
            pd.Series(current).value_counts().reindex(categories, fill_value=0).to_numpy(),
        ]
    )
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:
        return StatTest(statistic=0.0, p_value=1.0)
    chi2, p_value, _, _ = stats.chi2_contingency(table)
    cramers_v = float(np.sqrt(chi2 / table.sum()))  # tabela 2 x k: min(r - 1, c - 1) = 1
    return StatTest(statistic=cramers_v, p_value=float(p_value))


def two_proportion_test(successes_ref: int, n_ref: int, successes_cur: int, n_cur: int) -> StatTest:
    """Teste z para diferenca de proporcoes (ex.: taxa de inadimplencia); ``statistic`` e a diferenca."""
    p_ref, p_cur = successes_ref / n_ref, successes_cur / n_cur
    pooled = (successes_ref + successes_cur) / (n_ref + n_cur)
    std_error = np.sqrt(pooled * (1 - pooled) * (1 / n_ref + 1 / n_cur))
    if std_error == 0:
        return StatTest(statistic=float(p_cur - p_ref), p_value=1.0)
    z = (p_cur - p_ref) / std_error
    return StatTest(statistic=float(p_cur - p_ref), p_value=float(2 * stats.norm.sf(abs(z))))
