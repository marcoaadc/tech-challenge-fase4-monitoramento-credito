"""Analise causal da degradacao do modelo.

Duas abordagens complementares:

1. **Intervencional** (``mechanism_attribution``): como o cenario e um modelo causal conhecido,
   cada mecanismo pode ser ligado ou desligado isoladamente (o operador *do* de Pearl). Avaliam-se
   as 2^4 combinacoes e a variacao de cada metrica e repartida entre os mecanismos por valores de
   Shapley, que somam exatamente o efeito total. O experimento e repetido com varias seeds e os
   cenarios de uma mesma seed compartilham o ruido aleatorio. Responde: *o que causou a queda?*
2. **Observacional** (``calibration_scan`` e ``segment_diagnosis``): o que uma equipe conseguiria
   ver em producao, sem acesso ao gerador. A varredura compara inadimplencia observada e prevista
   em faixas de cada feature, com cortes tirados da propria referencia, e aponta onde o erro se
   concentrou. Responde: *onde procurar a causa?*
"""

from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd

from creditwatch.data.schema import SCORE, TARGET
from creditwatch.drift.detector import DriftThresholds, detect_drift
from creditwatch.drift.statistics import psi_numerical
from creditwatch.features.engineering import add_engineered_features
from creditwatch.model.metrics import classification_metrics
from creditwatch.model.registry import CreditScorer
from creditwatch.simulation.scenario import MECHANISMS, MonthPlan, generate_batch

ATTRIBUTED_METRICS = ("roc_auc", "ks", "default_rate", "bad_rate_approved", "approval_rate", "score_psi", "n_drifted")
SEGMENTS = ("em dia, utilização baixa", "em dia, utilização alta", "em atraso")
UTILIZATION = "utilization_1"
RECENT_STATUS = "pay_status_1"
MAX_CATEGORIES = 12
MIN_BIN_ROWS = 150


def evaluate_plan(
    pool: pd.DataFrame,
    plan: MonthPlan,
    scorer: CreditScorer,
    reference_full: pd.DataFrame,
    reference_holdout: pd.DataFrame,
    n: int,
    seed: int,
    thresholds: DriftThresholds | None = None,
) -> dict[str, float]:
    """Gera um lote sob ``plan`` e mede as metricas de modelo e de drift."""
    batch = add_engineered_features(generate_batch(pool, plan, n, seed))
    score = scorer.score(batch)
    metrics = classification_metrics(batch[TARGET].to_numpy(), score, scorer.threshold)
    drift = detect_drift(reference_full, batch, thresholds)
    return {
        **metrics,
        "score_psi": psi_numerical(reference_holdout[SCORE], score),
        "n_drifted": float(len(drift.drifted_features)),
        "share_drifted": drift.share_drifted,
    }


def shapley_values(values: dict[frozenset, float], players: tuple[str, ...]) -> dict[str, float]:
    """Valores de Shapley exatos de um jogo ``v(S)`` definido para todas as coalizoes de ``players``."""
    n = len(players)
    attribution: dict[str, float] = {}
    for player in players:
        others = [p for p in players if p != player]
        total = 0.0
        for size in range(n):
            weight = factorial(size) * factorial(n - size - 1) / factorial(n)
            for coalition in combinations(others, size):
                subset = frozenset(coalition)
                total += weight * (values[subset | {player}] - values[subset])
        attribution[player] = total
    return attribution


def _attribution_for_seed(
    pool: pd.DataFrame,
    plan: MonthPlan,
    active: tuple[str, ...],
    scorer: CreditScorer,
    reference_full: pd.DataFrame,
    reference_holdout: pd.DataFrame,
    n: int,
    seed: int,
    thresholds: DriftThresholds | None,
) -> dict:
    results: dict[frozenset, dict[str, float]] = {}
    for size in range(len(active) + 1):
        for coalition in combinations(active, size):
            results[frozenset(coalition)] = evaluate_plan(
                pool, plan.only(*coalition), scorer, reference_full, reference_holdout, n, seed, thresholds
            )
    baseline, full = results[frozenset()], results[frozenset(active)]
    return {
        "baseline": {metric: baseline[metric] for metric in ATTRIBUTED_METRICS},
        "full": {metric: full[metric] for metric in ATTRIBUTED_METRICS},
        "single": {
            name: {metric: results[frozenset({name})][metric] - baseline[metric] for metric in ATTRIBUTED_METRICS}
            for name in active
        },
        "shapley": {
            metric: shapley_values({subset: values[metric] for subset, values in results.items()}, active)
            for metric in ATTRIBUTED_METRICS
        },
    }


def _mean(items: list):
    """Media, campo a campo, de dicionarios aninhados com a mesma estrutura."""
    first = items[0]
    if isinstance(first, dict):
        return {key: _mean([item[key] for item in items]) for key in first}
    return float(np.mean(items))


def mechanism_attribution(
    pool: pd.DataFrame,
    plan: MonthPlan,
    scorer: CreditScorer,
    reference_full: pd.DataFrame,
    reference_holdout: pd.DataFrame,
    n: int = 2000,
    seeds: tuple[int, ...] = (42,),
    thresholds: DriftThresholds | None = None,
) -> dict:
    """Reparte a variacao de cada metrica entre os mecanismos ativos em ``plan``.

    Para cada seed, todas as combinacoes de mecanismos sao avaliadas em lotes de ``n`` propostas
    (o mesmo tamanho das remessas de producao, para que a intensidade do deslocamento seja a
    mesma). Devolve a media entre as seeds das metricas sem nenhum mecanismo (``baseline``), com
    todos (``full``), com cada um isolado (``single``) e da atribuicao de Shapley (``shapley``),
    alem do desvio-padrao da atribuicao entre as seeds (``shapley_std``).
    """
    active = tuple(name for name in MECHANISMS if getattr(plan, name) > 0)
    n = min(n, len(pool))
    runs = [
        _attribution_for_seed(pool, plan, active, scorer, reference_full, reference_holdout, n, seed, thresholds)
        for seed in seeds
    ]
    averaged = _mean(runs)
    return {
        "plan": plan.as_dict(),
        "sample_size": n,
        "seeds": list(seeds),
        "mechanisms": list(active),
        **averaged,
        "total_effect": {
            metric: averaged["full"][metric] - averaged["baseline"][metric] for metric in ATTRIBUTED_METRICS
        },
        "shapley_std": {
            metric: {name: float(np.std([run["shapley"][metric][name] for run in runs])) for name in active}
            for metric in ATTRIBUTED_METRICS
        },
    }


def utilization_cut(reference: pd.DataFrame) -> float:
    """Corte de utilizacao do limite para a segmentacao: a mediana da referencia."""
    return float(reference[UTILIZATION].median())


def assign_segment(frame: pd.DataFrame, cut: float) -> pd.Series:
    """Segmenta por comportamento de pagamento: em dia (utilizacao baixa ou alta) ou em atraso.

    ``cut`` vem dos dados (``utilization_cut``), nao do simulador: e a segmentacao que um analista
    montaria com as duas variaveis que a varredura de calibracao aponta.
    """
    on_time = frame[RECENT_STATUS] <= 0
    high = frame[UTILIZATION] >= cut
    labels = np.where(~on_time, SEGMENTS[2], np.where(high, SEGMENTS[1], SEGMENTS[0]))
    return pd.Series(labels, index=frame.index, name="segment")


def segment_diagnosis(periods: dict[str, pd.DataFrame], cut: float) -> list[dict]:
    """Inadimplencia observada x PD prevista por segmento, em cada periodo.

    Um segmento em que a inadimplencia observada descola da prevista, sem que suas features tenham
    mudado, indica que a relacao entre features e inadimplencia mudou ali.
    """
    rows: list[dict] = []
    for period, frame in periods.items():
        segments = assign_segment(frame, cut)
        for segment in SEGMENTS:
            part = frame[segments == segment]
            if part.empty:
                continue
            observed, predicted = float(part[TARGET].mean()), float(part[SCORE].mean())
            rows.append(
                {
                    "period": period,
                    "segment": segment,
                    "n": len(part),
                    "share": round(len(part) / len(frame), 4),
                    "observed_default_rate": round(observed, 4),
                    "predicted_default_rate": round(predicted, 4),
                    "gap": round(observed - predicted, 4),
                }
            )
    return rows


def _bins(reference: pd.Series, current: pd.Series, n_bins: int) -> tuple[pd.Series, pd.Series]:
    """Faixas de uma feature: categorias, se forem poucas, ou quantis da referencia."""
    if reference.nunique() <= MAX_CATEGORIES:
        return reference.astype(str), current.astype(str)
    inner = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)[1:-1]))
    edges = np.concatenate(([-np.inf], inner, [np.inf]))
    labels = [f"faixa {index + 1} de {len(edges) - 1}" for index in range(len(edges) - 1)]
    return (
        pd.cut(reference, edges, labels=labels).astype(str),
        pd.cut(current, edges, labels=labels).astype(str),
    )


def calibration_scan(
    reference: pd.DataFrame, current: pd.DataFrame, features: tuple[str, ...], n_bins: int = 4
) -> dict:
    """Procura as features em cujas faixas o erro de calibracao mais se concentrou.

    Para cada faixa de cada feature mede-se o *excesso* de erro: (observada - prevista) no periodo
    atual menos o mesmo valor na referencia. Se o modelo errasse por igual em toda a carteira, o
    excesso seria parecido em todas as faixas. A ``spread`` de uma feature e a diferenca entre o
    maior e o menor excesso das suas faixas: quanto maior, mais aquela feature separa quem o
    modelo passou a errar de quem ele continua acertando.

    Nenhum corte vem do simulador: categorias ou quartis da referencia.
    """
    rows: list[dict] = []
    for feature in features:
        ref_bins, cur_bins = _bins(reference[feature], current[feature], n_bins)
        for label in sorted(set(ref_bins) & set(cur_bins)):
            ref_part, cur_part = reference[ref_bins == label], current[cur_bins == label]
            if len(ref_part) < MIN_BIN_ROWS or len(cur_part) < MIN_BIN_ROWS:
                continue
            ref_gap = float(ref_part[TARGET].mean() - ref_part[SCORE].mean())
            cur_gap = float(cur_part[TARGET].mean() - cur_part[SCORE].mean())
            rows.append(
                {
                    "feature": feature,
                    "bin": label,
                    "n": len(cur_part),
                    "share": round(len(cur_part) / len(current), 4),
                    "reference_gap": round(ref_gap, 4),
                    "current_gap": round(cur_gap, 4),
                    "excess_gap": round(cur_gap - ref_gap, 4),
                }
            )
    by_feature: list[dict] = []
    for feature in features:
        bins = [row for row in rows if row["feature"] == feature]
        if len(bins) < 2:
            continue
        worst = max(bins, key=lambda row: row["excess_gap"])
        by_feature.append(
            {
                "feature": feature,
                "spread": round(worst["excess_gap"] - min(row["excess_gap"] for row in bins), 4),
                "worst_bin": worst["bin"],
                "worst_excess_gap": worst["excess_gap"],
                "worst_bin_share": worst["share"],
            }
        )
    return {
        "n_bins": n_bins,
        "by_feature": sorted(by_feature, key=lambda row: -row["spread"]),
        "bins": sorted(rows, key=lambda row: -row["excess_gap"]),
    }


def root_cause_ranking(feature_drift: list[dict], importance: dict[str, float], top: int = 10) -> list[dict]:
    """Ordena as features por PSI x importancia: candidatas a causa do drift de predicao."""
    ranked = [
        {
            "feature": feature["feature"],
            "psi": feature["psi"],
            "importance": importance.get(feature["feature"], 0.0),
            "impact": round(feature["psi"] * importance.get(feature["feature"], 0.0), 5),
            "severity": feature["severity"],
        }
        for feature in feature_drift
        if feature["feature"] in importance
    ]
    return sorted(ranked, key=lambda row: -row["impact"])[:top]
