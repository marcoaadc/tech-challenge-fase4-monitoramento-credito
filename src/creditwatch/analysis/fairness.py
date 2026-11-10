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
Z_95 = 1.959964
ADVERSE, INCONCLUSIVE, WITHIN_LIMIT = "impacto adverso", "inconclusivo", "dentro do limite"


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


def ratio_confidence_interval(p_group: float, n_group: int, p_best: float, n_best: int) -> tuple[float, float]:
    """Intervalo de 95% para a razao entre duas taxas de aprovacao (metodo do logaritmo).

    Trata os dois grupos como amostras independentes. Nos periodos de producao o mesmo cliente do
    pool pode aparecer em mais de um mes, entao o intervalo real e um pouco mais largo.
    """
    if min(p_group, p_best) <= 0 or min(n_group, n_best) <= 0:
        return float("nan"), float("nan")
    ratio = p_group / p_best
    std_error = np.sqrt((1 - p_group) / (n_group * p_group) + (1 - p_best) / (n_best * p_best))
    return float(ratio * np.exp(-Z_95 * std_error)), float(ratio * np.exp(Z_95 * std_error))


def _verdict(ratio: float, ci_low: float) -> str:
    if ratio < FOUR_FIFTHS:
        return ADVERSE
    return WITHIN_LIMIT if ci_low >= FOUR_FIFTHS else INCONCLUSIVE


def group_metrics(frame: pd.DataFrame, attribute: str) -> list[dict]:
    """Metricas de decisao por grupo de ``attribute``.

    Grupos abaixo de ``minimum_group_size`` continuam na saida, marcados com ``compared=False``,
    mas nao entram na comparacao nem podem ser o grupo de referencia.
    """
    rows: list[dict] = []
    minimum = minimum_group_size(len(frame))
    for group, part in frame.groupby(attribute, sort=True):
        good = part[part[TARGET] == 0]
        rows.append(
            {
                "attribute": attribute,
                "group": str(group),
                "n": len(part),
                "compared": len(part) >= minimum,
                "approval_rate": float(part[DECISION].mean()),
                "approval_rate_good_payers": float(good[DECISION].mean()) if len(good) else float("nan"),
                "default_rate": float(part[TARGET].mean()),
                "mean_score": float(part[SCORE].mean()),
            }
        )
    compared = [row for row in rows if row["compared"]]
    if compared:
        best = max(compared, key=lambda row: row["approval_rate"])
        best_good = np.nanmax([row["approval_rate_good_payers"] for row in compared])
        for row in compared:
            ratio = row["approval_rate"] / best["approval_rate"] if best["approval_rate"] else float("nan")
            if row is best:
                ci_low = ci_high = 1.0
            else:
                ci_low, ci_high = ratio_confidence_interval(
                    row["approval_rate"], row["n"], best["approval_rate"], best["n"]
                )
            row["disparate_impact"] = ratio
            row["disparate_impact_ci"] = [ci_low, ci_high]
            row["equal_opportunity_gap"] = float(best_good - row["approval_rate_good_payers"])
            row["adverse_impact"] = bool(ratio < FOUR_FIFTHS)
            row["verdict"] = _verdict(ratio, ci_low)
    return rows


def fairness_audit(periods: dict[str, pd.DataFrame]) -> dict:
    """Auditoria por periodo e atributo, com o resumo do grupo menos aprovado de cada atributo."""
    by_period: dict[str, list[dict]] = {}
    for period, frame in periods.items():
        labeled = with_group_labels(frame)
        by_period[period] = [row for attribute in ATTRIBUTES for row in group_metrics(labeled, attribute)]

    worst: list[dict] = []
    for period, rows in by_period.items():
        for attribute in ATTRIBUTES:
            candidates = [row for row in rows if row["attribute"] == attribute and row["compared"]]
            if candidates:
                lowest = min(candidates, key=lambda row: row["disparate_impact"])
                worst.append(
                    {
                        "period": period,
                        "attribute": attribute,
                        "group": lowest["group"],
                        "n": lowest["n"],
                        "disparate_impact": round(lowest["disparate_impact"], 4),
                        "disparate_impact_ci": [round(value, 4) for value in lowest["disparate_impact_ci"]],
                        "equal_opportunity_gap": round(lowest["equal_opportunity_gap"], 4),
                        "adverse_impact": lowest["adverse_impact"],
                        "verdict": lowest["verdict"],
                    }
                )
    compared = [{"period": period, **row} for period, rows in by_period.items() for row in rows if row["compared"]]
    largest_gap = max(compared, key=lambda row: row["equal_opportunity_gap"]) if compared else None
    return {
        "four_fifths_threshold": FOUR_FIFTHS,
        "minimum_group_size": {period: minimum_group_size(len(frame)) for period, frame in periods.items()},
        "attributes": list(ATTRIBUTES),
        "largest_equal_opportunity_gap": largest_gap
        and {
            "period": largest_gap["period"],
            "attribute": largest_gap["attribute"],
            "group": largest_gap["group"],
            "equal_opportunity_gap": round(largest_gap["equal_opportunity_gap"], 4),
        },
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
