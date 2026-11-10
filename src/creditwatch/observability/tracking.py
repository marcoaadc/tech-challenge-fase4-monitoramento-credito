"""Rastreamento no MLflow (ciclo de vida do modelo e historico do monitoramento).

A classe e um *no-op* quando nenhuma URI e configurada: o pipeline funciona sem o MLflow e os
testes nao dependem dele. Com a URI definida:

- experimento de treino: um run com parametros, metricas por candidato e o modelo registrado;
- experimento de monitoramento: um run pai por execucao, com as metricas de cada lote logadas
  com ``step`` (o MLflow desenha a serie temporal), e um run filho por lote com os relatorios.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger("creditwatch.observability.tracking")

_INVALID_KEY = re.compile(r"[^0-9A-Za-z_\-. /]")


def _clean_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Mantem apenas valores numericos finitos e normaliza os nomes para o MLflow."""
    cleaned: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            value = float(value)
        if isinstance(value, (int, float)) and math.isfinite(value):
            cleaned[_INVALID_KEY.sub("_", key)] = float(value)
    return cleaned


class Tracker:
    """Fachada fina sobre o MLflow, inerte quando ``tracking_uri`` e ``None``."""

    def __init__(self, tracking_uri: str | None, experiment: str):
        self.enabled = bool(tracking_uri)
        self.experiment = experiment
        self._mlflow = None
        if self.enabled:
            import mlflow

            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment)
            self._mlflow = mlflow
            logger.info("rastreamento MLflow ativo: %s (experimento %s)", tracking_uri, experiment)

    @contextmanager
    def run(self, name: str, nested: bool = False, tags: dict[str, str] | None = None) -> Iterator[Any]:
        if not self.enabled:
            yield None
            return
        with self._mlflow.start_run(run_name=name, nested=nested, tags=tags) as active:
            yield active

    def log_params(self, params: dict[str, Any]) -> None:
        if self.enabled:
            self._mlflow.log_params({key: str(value)[:500] for key, value in params.items()})

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self.enabled:
            cleaned = _clean_metrics(metrics)
            if cleaned:
                self._mlflow.log_metrics(cleaned, step=step)

    def set_tags(self, tags: dict[str, Any]) -> None:
        if self.enabled:
            self._mlflow.set_tags({key: str(value) for key, value in tags.items()})

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        if self.enabled and Path(path).exists():
            self._mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_dict(self, payload: dict, artifact_file: str) -> None:
        if self.enabled:
            self._mlflow.log_dict(payload, artifact_file)

    def log_model(self, model: Any, input_example: Any, registered_name: str | None = None) -> None:
        """Registra o modelo com assinatura inferida e, opcionalmente, no Model Registry.

        Cada tipo usa o seu formato nativo, sem pickle: XGBoost em UBJ (``mlflow.xgboost``) e
        pipelines scikit-learn em skops (``mlflow.sklearn``), que so carrega tipos confiaveis.
        """
        if not self.enabled:
            return
        from mlflow.models import infer_signature

        signature = infer_signature(input_example, model.predict_proba(input_example)[:, 1])
        common = {
            "name": "model",
            "signature": signature,
            "input_example": input_example.head(5),
            "registered_model_name": registered_name,
        }
        if type(model).__module__.startswith("xgboost"):
            self._mlflow.xgboost.log_model(model, **common)
        else:
            self._mlflow.sklearn.log_model(model, pyfunc_predict_fn="predict_proba", **common)


def export_tracking_summary(tracking_uri: str | None, experiments: tuple[str, ...], path: Path) -> Path | None:
    """Grava um resumo do que esta no MLflow: experimentos, runs, contagem de metricas e modelos registrados.

    O banco do MLflow nao e versionado; este arquivo e a evidencia versionada de que o
    rastreamento aconteceu e do que ele contem. Sem ``tracking_uri`` nada e gravado.
    """
    if not tracking_uri:
        return None
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    summary: dict = {"tracking_uri": tracking_uri, "experiments": [], "registered_models": []}
    for name in experiments:
        experiment = client.get_experiment_by_name(name)
        if experiment is None:
            continue
        runs = client.search_runs([experiment.experiment_id], order_by=["attributes.start_time ASC"])
        summary["experiments"].append(
            {
                "name": name,
                "runs": len(runs),
                "parent_runs": sum("mlflow.parentRunId" not in run.data.tags for run in runs),
                "items": [
                    {
                        "run_name": run.info.run_name,
                        "nested": "mlflow.parentRunId" in run.data.tags,
                        "status": run.info.status,
                        "metrics": len(run.data.metrics),
                        "params": len(run.data.params),
                    }
                    for run in runs
                ],
            }
        )
    for model in client.search_registered_models():
        versions = client.search_model_versions(f"name='{model.name}'")
        summary["registered_models"].append(
            {"name": model.name, "versions": sorted(int(version.version) for version in versions)}
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    path.write_text(payload, encoding="utf-8", newline="\n")
    logger.info("resumo do rastreamento MLflow gravado em %s", path)
    return path
