"""Metricas de discriminacao, calibracao e de negocio de um modelo de credit scoring."""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def ks_statistic(y_true: np.ndarray, score: np.ndarray) -> float:
    """KS do modelo: maior separacao entre as distribuicoes de score de bons e maus pagadores."""
    y_true = np.asarray(y_true)
    score = np.asarray(score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(stats.ks_2samp(score[y_true == 1], score[y_true == 0]).statistic)


def best_ks_threshold(y_true: np.ndarray, score: np.ndarray) -> float:
    """Ponto de corte que maximiza TPR - FPR (indice J de Youden), usado como politica de aprovacao."""
    fpr, tpr, thresholds = roc_curve(y_true, score)
    return float(thresholds[int(np.argmax(tpr - fpr))])


def classification_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    """Metricas de um lote rotulado. ``score`` e a probabilidade de inadimplencia (PD).

    A decisao e ``aprovar se PD < threshold``. ``bad_rate_approved`` (inadimplencia entre os
    aprovados) e a metrica que traduz a degradacao do modelo em perda financeira.
    """
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    rejected = (score >= threshold).astype(int)
    approved = rejected == 0
    has_both_classes = len(np.unique(y_true)) == 2
    auc = float(roc_auc_score(y_true, score)) if has_both_classes else float("nan")
    return {
        "n": len(y_true),
        "roc_auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_statistic(y_true, score),
        "pr_auc": float(average_precision_score(y_true, score)) if has_both_classes else float("nan"),
        "brier": float(brier_score_loss(y_true, score)),
        "precision": float(precision_score(y_true, rejected, zero_division=0)),
        "recall": float(recall_score(y_true, rejected, zero_division=0)),
        "f1": float(f1_score(y_true, rejected, zero_division=0)),
        "default_rate": float(y_true.mean()),
        "mean_score": float(score.mean()),
        "calibration_gap": float(y_true.mean() - score.mean()),
        "approval_rate": float(approved.mean()),
        "bad_rate_approved": float(y_true[approved].mean()) if approved.any() else float("nan"),
    }


def score_summary(score: np.ndarray, threshold: float) -> dict[str, float]:
    """Metricas disponiveis sem rotulo (no momento da decisao)."""
    score = np.asarray(score, dtype=float)
    return {
        "n": len(score),
        "mean_score": float(score.mean()),
        "approval_rate": float((score < threshold).mean()),
    }
