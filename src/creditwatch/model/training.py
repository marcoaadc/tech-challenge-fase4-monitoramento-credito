"""Treino do modelo baseline de risco de credito.

Dois candidatos sao treinados no Dataset de Referencia (particao ``train``): uma Regressao
Logistica (baseline interpretavel) e um XGBoost. Vence o de maior ROC AUC na validacao; o ponto de
corte da politica de aprovacao maximiza o KS na validacao e as metricas finais sao medidas no teste.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from creditwatch.data.schema import MODEL_FEATURES, TARGET
from creditwatch.data.split import SPLIT_COLUMN
from creditwatch.features.engineering import model_matrix
from creditwatch.model.metrics import best_ks_threshold, classification_metrics

logger = logging.getLogger("creditwatch.model.training")

XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "eval_metric": "auc",
}
LOGREG_PARAMS = {"C": 1.0, "max_iter": 2000}
CANDIDATES = ("logreg", "xgboost")


def build_candidate(name: str, seed: int = 42) -> BaseEstimator:
    if name == "logreg":
        return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(random_state=seed, **LOGREG_PARAMS))])
    if name == "xgboost":
        return XGBClassifier(random_state=seed, n_jobs=1, **XGBOOST_PARAMS)
    raise ValueError(f"candidato desconhecido: {name!r}; opcoes: {CANDIDATES}")


def candidate_params(name: str) -> dict:
    return dict(XGBOOST_PARAMS if name == "xgboost" else LOGREG_PARAMS)


def predict_default_probability(model: BaseEstimator, frame: pd.DataFrame) -> np.ndarray:
    """Probabilidade de inadimplencia (PD) para cada linha de ``frame``."""
    return model.predict_proba(model_matrix(frame))[:, 1]


def feature_importance(model: BaseEstimator) -> dict[str, float]:
    """Importancia normalizada das features (ganho no XGBoost, |coeficiente| na regressao)."""
    if isinstance(model, Pipeline):
        raw = np.abs(model.named_steps["clf"].coef_[0])
    else:
        raw = np.asarray(model.feature_importances_, dtype=float)
    total = raw.sum()
    normalized = raw / total if total > 0 else raw
    ranked = sorted(zip(MODEL_FEATURES, normalized, strict=True), key=lambda item: -item[1])
    return {name: round(float(value), 5) for name, value in ranked}


@dataclass
class CandidateResult:
    name: str
    model: BaseEstimator
    threshold: float
    val_metrics: dict[str, float]
    fit_seconds: float


@dataclass
class TrainingResult:
    selected: CandidateResult
    candidates: list[CandidateResult]
    test_metrics: dict[str, float]
    importance: dict[str, float]
    split_sizes: dict[str, int] = field(default_factory=dict)

    @property
    def model(self) -> BaseEstimator:
        return self.selected.model

    @property
    def threshold(self) -> float:
        return self.selected.threshold


def train_and_select(
    reference: pd.DataFrame, seed: int = 42, candidates: tuple[str, ...] = CANDIDATES
) -> TrainingResult:
    """Treina os candidatos na referencia e seleciona o melhor por ROC AUC de validacao."""
    parts = {name: reference[reference[SPLIT_COLUMN] == name] for name in ("train", "val", "test")}
    if any(part.empty for part in parts.values()):
        raise ValueError("a referencia precisa das particoes train, val e test (use assign_model_splits)")

    results: list[CandidateResult] = []
    for name in candidates:
        model = build_candidate(name, seed)
        started = time.perf_counter()
        model.fit(model_matrix(parts["train"]), parts["train"][TARGET])
        fit_seconds = time.perf_counter() - started
        val_score = predict_default_probability(model, parts["val"])
        threshold = best_ks_threshold(parts["val"][TARGET].to_numpy(), val_score)
        metrics = classification_metrics(parts["val"][TARGET].to_numpy(), val_score, threshold)
        results.append(CandidateResult(name, model, threshold, metrics, fit_seconds))
        logger.info("candidato %s: roc_auc(val)=%.4f ks(val)=%.4f", name, metrics["roc_auc"], metrics["ks"])

    selected = max(results, key=lambda result: result.val_metrics["roc_auc"])
    test_score = predict_default_probability(selected.model, parts["test"])
    test_metrics = classification_metrics(parts["test"][TARGET].to_numpy(), test_score, selected.threshold)
    logger.info(
        "selecionado %s: roc_auc(teste)=%.4f ks(teste)=%.4f", selected.name, test_metrics["roc_auc"], test_metrics["ks"]
    )
    return TrainingResult(
        selected=selected,
        candidates=results,
        test_metrics=test_metrics,
        importance=feature_importance(selected.model),
        split_sizes={name: len(part) for name, part in parts.items()},
    )
