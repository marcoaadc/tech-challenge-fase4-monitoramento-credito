"""Estagio 2 - treino do modelo baseline no Dataset de Referencia."""

from __future__ import annotations

import logging

from creditwatch.config.settings import Settings, get_settings
from creditwatch.data.split import SPLIT_COLUMN
from creditwatch.features.engineering import model_matrix
from creditwatch.model.registry import METADATA_FILENAME, ModelMetadata, save_model
from creditwatch.model.training import CANDIDATES, candidate_params, train_and_select
from creditwatch.observability.events import StageTracker
from creditwatch.observability.tracking import Tracker
from creditwatch.pipelines._cli import build_parser, read_table, start_run, write_json

logger = logging.getLogger("creditwatch.pipelines.train")


def run_train(
    settings: Settings, tracker: StageTracker | None = None, candidates: tuple[str, ...] = CANDIDATES
) -> ModelMetadata:
    tracker = tracker or StageTracker("train", settings.events_path)
    mlflow_tracker = Tracker(settings.mlflow_tracking_uri, settings.mlflow_experiment_training)

    with tracker.stage("load_reference") as event:
        reference = read_table(settings.reference_path)
        event.rows_out = len(reference)

    with tracker.stage("fit_candidates", rows_in=len(reference)) as event:
        result = train_and_select(reference, seed=settings.seed, candidates=candidates)
        event.details = {"selected": result.selected.name, "roc_auc_test": round(result.test_metrics["roc_auc"], 4)}

    with tracker.stage("save_model") as event:
        metadata = save_model(result, settings.reference_path, settings.models_dir, settings.seed)
        event.details = {"model_version": metadata.model_version}

    report = {
        "model_version": metadata.model_version,
        "selected": result.selected.name,
        "threshold": metadata.threshold,
        "candidates": [
            {
                "name": c.name,
                "fit_seconds": round(c.fit_seconds, 2),
                "validation": metadata.metrics["validation"][c.name],
            }
            for c in result.candidates
        ],
        "test": metadata.metrics["test"],
        "feature_importance": metadata.feature_importance,
        "splits": result.split_sizes,
    }
    write_json(settings.reports_dir / "model" / "training_metrics.json", report)

    with mlflow_tracker.run(f"treino-{metadata.model_version}", tags={"stage": "training"}):
        mlflow_tracker.log_params(
            {
                "model_type": metadata.model_type,
                "model_version": metadata.model_version,
                "threshold": metadata.threshold,
                "reference_sha256": metadata.reference["sha256"],
                "n_features": len(metadata.features),
                **metadata.params,
            }
        )
        mlflow_tracker.log_metrics({f"test_{key}": value for key, value in result.test_metrics.items()})
        for candidate in result.candidates:
            with mlflow_tracker.run(candidate.name, nested=True, tags={"candidate": candidate.name}):
                mlflow_tracker.log_params(candidate_params(candidate.name))
                mlflow_tracker.log_metrics({f"val_{key}": value for key, value in candidate.val_metrics.items()})
                mlflow_tracker.log_metrics({"fit_seconds": candidate.fit_seconds})
        mlflow_tracker.log_artifact(settings.models_dir / METADATA_FILENAME)
        train_rows = reference[reference[SPLIT_COLUMN] == "train"]
        mlflow_tracker.log_model(result.model, model_matrix(train_rows.head(200)), settings.mlflow_registered_model)
    return metadata


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    parser = build_parser("Treina os candidatos, seleciona o melhor e grava modelo + metadados.")
    args = parser.parse_args(argv)
    start_run(settings, args.log_level)
    run_train(settings)


if __name__ == "__main__":
    main()
