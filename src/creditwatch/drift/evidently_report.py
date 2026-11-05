"""Relatorio de drift do Evidently: Dataset de Referencia x lote de producao.

O Evidently gera o relatorio HTML navegavel (distribuicoes lado a lado, teste por coluna) e um
veredito independente do detector proprio (``creditwatch.drift.detector``). Os dois usam PSI com
os mesmos limiares; divergencias pontuais vem apenas da forma de montar as faixas e servem de
contraprova. Quando o lote ja tem rotulo, o relatorio inclui a qualidade da classificacao.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from creditwatch.data.schema import CATEGORICAL_MONITORED, NUMERICAL_MONITORED, SCORE, TARGET

PSI_METHOD = "psi"


def _columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numerical = [c for c in (*NUMERICAL_MONITORED, SCORE) if c in frame.columns]
    categorical = [c for c in CATEGORICAL_MONITORED if c in frame.columns]
    return numerical, categorical


def run_evidently_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    psi_threshold: float = 0.10,
    drift_share: float = 0.30,
    decision_threshold: float | None = None,
    batch: str | None = None,
):
    """Executa o relatorio e devolve o ``Snapshot`` do Evidently.

    ``decision_threshold`` liga a secao de qualidade da classificacao (exige ``default`` e ``score``
    nos dois datasets).
    """
    from evidently import BinaryClassification, DataDefinition, Dataset, Report
    from evidently.presets import ClassificationPreset, DataDriftPreset

    numerical, categorical = _columns(current)
    with_quality = decision_threshold is not None and TARGET in current.columns and TARGET in reference.columns
    keep = [*numerical, *categorical, *([TARGET] if with_quality else [])]
    definition = DataDefinition(
        numerical_columns=numerical,
        categorical_columns=categorical,
        classification=[BinaryClassification(target=TARGET, prediction_probas=SCORE)] if with_quality else None,
    )
    metrics = [
        DataDriftPreset(
            columns=[*numerical, *categorical], method=PSI_METHOD, threshold=psi_threshold, drift_share=drift_share
        )
    ]
    if with_quality:
        metrics.append(ClassificationPreset(probas_threshold=decision_threshold))

    report = Report(metrics, include_tests=True, tags=[batch] if batch else None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return report.run(
            Dataset.from_pandas(current[keep], data_definition=definition),
            Dataset.from_pandas(reference[keep], data_definition=definition),
            name=f"creditwatch-{batch}" if batch else None,
        )


def save_html(snapshot, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(path))
    return path


def summarize(snapshot, psi_threshold: float = 0.10) -> dict:
    """Extrai do snapshot o veredito do Evidently: PSI por coluna e colunas com drift."""
    psi: dict[str, float] = {}
    share: float | None = None
    for metric in snapshot.dict()["metrics"]:
        config = metric.get("config", {})
        kind = str(config.get("type", ""))
        value = metric.get("value")
        if kind.endswith("ValueDrift") and isinstance(value, (int, float)):
            psi[config["column"]] = round(float(value), 5)
        elif kind.endswith("DriftedColumnsCount") and isinstance(value, dict):
            share = float(value.get("share", float("nan")))
    drifted = sorted(column for column, value in psi.items() if value >= psi_threshold)
    return {
        "method": PSI_METHOD,
        "threshold": psi_threshold,
        "n_columns": len(psi),
        "n_drifted": len(drifted),
        "share_drifted": round(share if share is not None else len(drifted) / max(len(psi), 1), 4),
        "drifted_columns": drifted,
        "psi": psi,
    }
