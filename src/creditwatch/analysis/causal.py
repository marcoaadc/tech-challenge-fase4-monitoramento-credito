"""Analise causal da degradacao do modelo.

Duas abordagens complementares:

1. **Intervencional** (``mechanism_attribution``): como o cenario e um modelo causal conhecido,
   cada mecanismo pode ser ligado ou desligado isoladamente (o operador *do* de Pearl). Avaliam-se
   as 2^4 combinacoes e a variacao de cada metrica e repartida entre os mecanismos por valores de
   Shapley, que somam exatamente o efeito total. Responde: *o que causou a queda de AUC?*
2. **Observacional** (``segment_diagnosis`` e ``root_cause_ranking``): o que uma equipe conseguiria
   ver em producao, sem acesso ao gerador. Compara inadimplencia observada e prevista por segmento
   e ordena as features por drift x importancia. Responde: *onde procurar a causa?*
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
from creditwatch.simulation.mechanisms import LEVERAGED_UTILIZATION
from creditwatch.simulation.scenario import MECHANISMS, MonthPlan, generate_batch

ATTRIBUTED_METRICS = ("roc_auc", "ks", "default_rate", "bad_rate_approved", "approval_rate", "score_psi", "n_drifted")
SEGMENTS = ("em dia, baixa utilização", "em dia, alavancado", "em atraso")


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


def mechanism_attribution(
    pool: pd.DataFrame,
    plan: MonthPlan,
    scorer: CreditScorer,
    reference_full: pd.DataFrame,
    reference_holdout: pd.DataFrame,
    n: int = 6000,
    seed: int = 42,
    thresholds: DriftThresholds | None = None,
) -> dict:
    """Reparte a variacao de cada metrica entre os mecanismos ativos em ``plan``.

    Devolve as metricas sem nenhum mecanismo (``baseline``), com todos (``full``), com cada um
    isolado (``single``) e a atribuicao de Shapley (``shapley``), cuja soma e ``full - baseline``.
    """
    active = tuple(name for name in MECHANISMS if getattr(plan, name) > 0)
    n = min(n, len(pool) // 2)
    results: dict[frozenset, dict[str, float]] = {}
    for size in range(len(active) + 1):
        for coalition in combinations(active, size):
            results[frozenset(coalition)] = evaluate_plan(
                pool, plan.only(*coalition), scorer, reference_full, reference_holdout, n, seed, thresholds
            )

    baseline, full = results[frozenset()], results[frozenset(active)]
    shapley = {
        metric: shapley_values({subset: values[metric] for subset, values in results.items()}, active)
        for metric in ATTRIBUTED_METRICS
    }
    return {
        "plan": plan.as_dict(),
        "sample_size": n,
        "mechanisms": list(active),
        "baseline": {metric: baseline[metric] for metric in ATTRIBUTED_METRICS},
        "full": {metric: full[metric] for metric in ATTRIBUTED_METRICS},
        "total_effect": {metric: full[metric] - baseline[metric] for metric in ATTRIBUTED_METRICS},
        "single": {
            name: {metric: results[frozenset({name})][metric] - baseline[metric] for metric in ATTRIBUTED_METRICS}
            for name in active
        },
        "shapley": shapley,
    }


def assign_segment(frame: pd.DataFrame) -> pd.Series:
    """Segmenta por comportamento de pagamento: em dia (baixa/alta utilizacao) ou em atraso."""
    on_time = frame["pay_status_1"] <= 0
    leveraged = frame["utilization_1"] >= LEVERAGED_UTILIZATION
    labels = np.where(~on_time, SEGMENTS[2], np.where(leveraged, SEGMENTS[1], SEGMENTS[0]))
    return pd.Series(labels, index=frame.index, name="segment")


def segment_diagnosis(periods: dict[str, pd.DataFrame]) -> list[dict]:
    """Inadimplencia observada x PD prevista por segmento, em cada periodo.

    Um segmento em que a inadimplencia observada descola da prevista, sem que suas features tenham
    mudado, e a assinatura de drift de conceito: a relacao P(y|x) mudou ali.
    """
    rows: list[dict] = []
    for period, frame in periods.items():
        segments = assign_segment(frame)
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
