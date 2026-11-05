"""Experimento de remediacao: o que recupera o modelo depois do drift?

Simula a decisao que a equipe tomaria ao fim do periodo, usando apenas rotulos ja maduros, e
compara estrategias em meses posteriores (*out-of-time*), nunca vistos por nenhuma delas:

- ``campeao``: manter o modelo atual (nao fazer nada);
- ``retreino_janela_recente``: retreinar so com os meses de producao mais recentes;
- ``retreino_referencia_mais_producao``: retreinar com a referencia mais toda a producao rotulada.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator

from creditwatch.data.schema import BATCH_MONTH, TARGET
from creditwatch.features.engineering import model_matrix
from creditwatch.model.metrics import best_ks_threshold, classification_metrics
from creditwatch.model.training import build_candidate, predict_default_probability

REPORTED = ("roc_auc", "ks", "brier", "calibration_gap", "approval_rate", "bad_rate_approved")
VALIDATION_FRACTION = 0.2


def _fit_challenger(train: pd.DataFrame, model_type: str, seed: int) -> tuple[BaseEstimator, float]:
    """Treina um desafiante e escolhe o ponto de corte em uma fatia de validacao."""
    validation = train.sample(frac=VALIDATION_FRACTION, random_state=seed)
    fit_rows = train.drop(validation.index)
    model = build_candidate(model_type, seed)
    model.fit(model_matrix(fit_rows), fit_rows[TARGET])
    threshold = best_ks_threshold(validation[TARGET].to_numpy(), predict_default_probability(model, validation))
    return model, threshold


def _evaluate(model: BaseEstimator, threshold: float, frame: pd.DataFrame) -> dict[str, float]:
    metrics = classification_metrics(frame[TARGET].to_numpy(), predict_default_probability(model, frame), threshold)
    return {key: round(metrics[key], 4) for key in REPORTED}


def remediation_experiment(
    champion: BaseEstimator,
    champion_threshold: float,
    model_type: str,
    reference_train: pd.DataFrame,
    production: pd.DataFrame,
    train_months: list[str],
    recent_months: list[str],
    test_months: list[str],
    seed: int = 42,
) -> dict:
    """Compara as estrategias nos ``test_months``.

    ``train_months`` sao os meses de producao com rotulo maduro na data da decisao;
    ``recent_months`` e o subconjunto mais recente deles (janela deslizante).
    """
    overlap = set(train_months) & set(test_months)
    if overlap:
        raise ValueError(f"meses de teste nao podem estar no treino: {sorted(overlap)}")
    labeled = production[production[BATCH_MONTH].isin(train_months)]
    recent = production[production[BATCH_MONTH].isin(recent_months)]
    test = production[production[BATCH_MONTH].isin(test_months)]
    if labeled.empty or recent.empty or test.empty:
        raise ValueError("faltam meses de producao para o experimento de remediacao")

    combined = pd.concat([reference_train, labeled], ignore_index=True)
    strategies = {
        "campeao": (champion, champion_threshold, len(reference_train)),
        "retreino_janela_recente": (*_fit_challenger(recent, model_type, seed), len(recent)),
        "retreino_referencia_mais_producao": (*_fit_challenger(combined, model_type, seed), len(combined)),
    }
    return {
        "model_type": model_type,
        "train_months": train_months,
        "recent_months": recent_months,
        "test_months": test_months,
        "test_rows": len(test),
        "strategies": {
            name: {
                "train_rows": int(rows),
                "threshold": round(float(threshold), 4),
                **_evaluate(model, threshold, test),
            }
            for name, (model, threshold, rows) in strategies.items()
        },
    }
