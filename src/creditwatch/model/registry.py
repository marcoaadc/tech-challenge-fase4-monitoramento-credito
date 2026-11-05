"""Persistencia do modelo e dos seus metadados (rastreabilidade ponta a ponta).

``metadata.json`` acompanha o artefato e registra versao, hiperparametros, metricas, ponto de
corte, versoes das bibliotecas e o SHA-256 do dataset de referencia e do proprio modelo: qualquer
decisao de credito pode ser rastreada ate o dado e o codigo que a produziram (LGPD, art. 20).
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import BaseEstimator

from creditwatch.data.schema import MODEL_FEATURES, PROTECTED
from creditwatch.data.source import sha256_of
from creditwatch.model.training import TrainingResult, candidate_params, predict_default_probability

MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"


@dataclass
class ModelMetadata:
    model_version: str
    model_type: str
    trained_at: str
    threshold: float
    features: list[str]
    excluded_protected_attributes: list[str]
    params: dict
    metrics: dict
    feature_importance: dict[str, float]
    reference: dict
    artifacts: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)

    def save(self, directory: Path) -> Path:
        path = Path(directory) / METADATA_FILENAME
        payload = json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n"
        path.write_text(payload, encoding="utf-8", newline="\n")
        return path

    @classmethod
    def load(cls, directory: Path) -> ModelMetadata:
        return cls(**json.loads((Path(directory) / METADATA_FILENAME).read_text(encoding="utf-8")))


def _rounded(metrics: dict[str, float]) -> dict[str, float]:
    return {key: (round(value, 5) if isinstance(value, float) else value) for key, value in metrics.items()}


def save_model(result: TrainingResult, reference_path: Path, models_dir: Path, seed: int) -> ModelMetadata:
    """Grava ``model.joblib`` + ``metadata.json`` em ``models_dir``."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / MODEL_FILENAME
    joblib.dump(result.model, model_path)

    reference_sha = sha256_of(reference_path)
    trained_at = datetime.now(timezone.utc).replace(microsecond=0)
    metadata = ModelMetadata(
        model_version=f"{trained_at:%Y%m%d}-{reference_sha[:8]}",
        model_type=result.selected.name,
        trained_at=trained_at.isoformat(),
        threshold=round(result.threshold, 6),
        features=list(MODEL_FEATURES),
        excluded_protected_attributes=list(PROTECTED),
        params={"seed": seed, **candidate_params(result.selected.name)},
        metrics={
            "validation": {c.name: _rounded(c.val_metrics) for c in result.candidates},
            "test": _rounded(result.test_metrics),
        },
        feature_importance=result.importance,
        reference={"path": Path(reference_path).as_posix(), "sha256": reference_sha, "splits": result.split_sizes},
        artifacts={"model": {"file": MODEL_FILENAME, "sha256": sha256_of(model_path)}},
        environment={
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    )
    metadata.save(models_dir)
    return metadata


class CreditScorer:
    """Modelo carregado + politica de decisao (``aprovar se PD < threshold``)."""

    def __init__(self, model: BaseEstimator, metadata: ModelMetadata):
        self.model = model
        self.metadata = metadata

    @classmethod
    def load(cls, models_dir: Path, verify_integrity: bool = True) -> CreditScorer:
        models_dir = Path(models_dir)
        metadata = ModelMetadata.load(models_dir)
        model_path = models_dir / MODEL_FILENAME
        if verify_integrity:
            expected = metadata.artifacts["model"]["sha256"]
            actual = sha256_of(model_path)
            if actual != expected:
                raise RuntimeError(f"artefato do modelo adulterado: sha256 {actual} != {expected}")
        return cls(joblib.load(model_path), metadata)

    @property
    def threshold(self) -> float:
        return self.metadata.threshold

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        return predict_default_probability(self.model, frame)

    def decide(self, score: np.ndarray) -> np.ndarray:
        return np.asarray(score) < self.threshold
