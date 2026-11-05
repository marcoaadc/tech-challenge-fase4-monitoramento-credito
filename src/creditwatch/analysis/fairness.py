"""Auditoria de vies: como as decisoes do modelo se distribuem entre grupos protegidos.

Os atributos sociodemograficos nao entram no modelo, mas isso nao garante tratamento igual: as
features financeiras podem atuar como *proxies*. A auditoria mede, por grupo:

- **taxa de aprovacao** e a razao de **impacto desigual** (regra dos quatro quintos: a taxa de um
  grupo abaixo de 80% da taxa do grupo mais aprovado sinaliza impacto adverso);
- **taxa de aprovacao entre bons pagadores** (igualdade de oportunidade): negar credito a quem
  pagaria e o erro que mais prejudica o titular;
- inadimplencia observada e PD media, para separar diferenca de risco real de vies do modelo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from creditwatch.data.schema import (
    DECISION,
    EDUCATION_DOMAIN,
    MARRIAGE_DOMAIN,
    MODEL_FEATURES,
    PROTECTED,
    SCORE,
    SEX_DOMAIN,
    TARGET,
)
from creditwatch.data.split import SPLIT_COLUMN
from creditwatch.features.engineering import add_engineered_features
from creditwatch.model.training import build_candidate

FOUR_FIFTHS = 0.80
AGE_BANDS = ((18, 29, "18-29"), (30, 39, "30-39"), (40, 49, "40-49"), (50, 120, "50+"))
ATTRIBUTES = ("sex", "age_band", "education", "marriage")
MIN_GROUP_SIZE = 100
MIN_GROUP_SHARE = 0.02


def age_band(age: pd.Series) -> pd.Series:
    bands = pd.Series("", index=age.index, dtype="object")
    for low, high, label in AGE_BANDS:
        bands[(age >= low) & (age <= high)] = label
    return bands


def with_group_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Adiciona colunas legiveis para os atributos protegidos."""
    out = frame.copy()
    out["sex"] = out["sex"].map(SEX_DOMAIN)
    out["education"] = out["education"].map(EDUCATION_DOMAIN)
    out["marriage"] = out["marriage"].map(MARRIAGE_DOMAIN)
    out["age_band"] = age_band(frame["age"])
    return out


def minimum_group_size(n_rows: int) -> int:
    """Tamanho minimo para um grupo entrar na comparacao: 100 linhas ou 2% da amostra, o que for maior.

    Um grupo pequeno tem taxa de aprovacao muito ruidosa (com 120 clientes, a margem de erro passa
    de 7 pontos percentuais) e, se virar o grupo de referencia, fabrica impacto adverso nos demais.
    """
    return max(MIN_GROUP_SIZE, int(MIN_GROUP_SHARE * n_rows))


def group_metrics(frame: pd.DataFrame, attribute: str) -> list[dict]:
    """Metricas de decisao por grupo de ``attribute``; grupos abaixo de ``minimum_group_size`` sao omitidos."""
    rows: list[dict] = []
    minimum = minimum_group_size(len(frame))
    for group, part in frame.groupby(attribute, sort=True):
        if len(part) < minimum:
            continue
        good = part[part[TARGET] == 0]
        rows.append(
            {
                "attribute": attribute,
                "group": str(group),
                "n": len(part),
                "approval_rate": float(part[DECISION].mean()),
                "approval_rate_good_payers": float(good[DECISION].mean()) if len(good) else float("nan"),
                "default_rate": float(part[TARGET].mean()),
                "mean_score": float(part[SCORE].mean()),
            }
        )
    if rows:
        best = max(row["approval_rate"] for row in rows)
        best_good = max(row["approval_rate_good_payers"] for row in rows)
        for row in rows:
            row["disparate_impact"] = row["approval_rate"] / best if best else float("nan")
            row["equal_opportunity_gap"] = best_good - row["approval_rate_good_payers"]
            row["adverse_impact"] = bool(row["disparate_impact"] < FOUR_FIFTHS)
    return rows


def fairness_audit(periods: dict[str, pd.DataFrame]) -> dict:
    """Auditoria por periodo e atributo, com o resumo dos grupos em impacto adverso."""
    by_period: dict[str, list[dict]] = {}
    for period, frame in periods.items():
        labeled = with_group_labels(frame)
        by_period[period] = [row for attribute in ATTRIBUTES for row in group_metrics(labeled, attribute)]

    worst: list[dict] = []
    for period, rows in by_period.items():
        for attribute in ATTRIBUTES:
            candidates = [row for row in rows if row["attribute"] == attribute]
            if candidates:
                lowest = min(candidates, key=lambda row: row["disparate_impact"])
                worst.append(
                    {
                        "period": period,
                        "attribute": attribute,
                        "group": lowest["group"],
                        "disparate_impact": round(lowest["disparate_impact"], 4),
                        "equal_opportunity_gap": round(lowest["equal_opportunity_gap"], 4),
                        "adverse_impact": lowest["adverse_impact"],
                    }
                )
    return {
        "four_fifths_threshold": FOUR_FIFTHS,
        "minimum_group_size": {period: minimum_group_size(len(frame)) for period, frame in periods.items()},
        "attributes": list(ATTRIBUTES),
        "worst_groups": worst,
        "by_period": by_period,
    }


def protected_attribute_ablation(
    reference: pd.DataFrame, model_type: str = "xgboost", seeds: tuple[int, ...] = (42, 7, 123)
) -> dict:
    """Mede o que o modelo ganharia se usasse os atributos protegidos como preditores.

    Treina o mesmo algoritmo com e sem ``PROTECTED`` (varias seeds) e compara o ROC AUC em
    validacao e teste. E a evidencia que sustenta a decisao de mante-los fora do modelo: se o ganho
    for nulo, exclui-los nao custa nada em desempenho.
    """
    enriched = add_engineered_features(reference)
    parts = {name: enriched[enriched[SPLIT_COLUMN] == name] for name in ("train", "val", "test")}
    variants = {"without_protected": list(MODEL_FEATURES), "with_protected": [*MODEL_FEATURES, *PROTECTED]}
    results: dict[str, dict[str, float]] = {}
    for name, columns in variants.items():
        scores = {"val": [], "test": []}
        for seed in seeds:
            model = build_candidate(model_type, seed)
            model.fit(parts["train"][columns].astype(float), parts["train"][TARGET])
            for split in scores:
                probability = model.predict_proba(parts[split][columns].astype(float))[:, 1]
                scores[split].append(roc_auc_score(parts[split][TARGET], probability))
        results[name] = {
            "roc_auc_val": round(float(np.mean(scores["val"])), 4),
            "roc_auc_test": round(float(np.mean(scores["test"])), 4),
            "roc_auc_test_std": round(float(np.std(scores["test"])), 4),
        }
    return {
        "model_type": model_type,
        "seeds": list(seeds),
        "protected_attributes": list(PROTECTED),
        **results,
        "gain_val": round(results["with_protected"]["roc_auc_val"] - results["without_protected"]["roc_auc_val"], 4),
        "gain_test": round(results["with_protected"]["roc_auc_test"] - results["without_protected"]["roc_auc_test"], 4),
    }
