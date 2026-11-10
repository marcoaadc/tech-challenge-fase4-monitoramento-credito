"""Pipeline completo: ingestao -> treino -> simulacao -> monitoramento -> analises -> dashboard.

Todos os estagios compartilham o mesmo ``run_id`` e o mesmo arquivo de eventos, de modo que uma
execucao inteira pode ser reconstruida a partir de ``reports/observability/pipeline_events.jsonl``.
Ao final, o log da execucao e o resumo do que foi rastreado no MLflow sao copiados para
``reports/observability/``: sao evidencias que nao contem dados pessoais e, ao contrario de
``logs/`` e de ``mlflow.db``, ficam versionadas.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from creditwatch.config.settings import Settings, get_settings
from creditwatch.observability.events import StageTracker
from creditwatch.observability.logs import LOG_FILENAME, current_run_id
from creditwatch.observability.tracking import export_tracking_summary
from creditwatch.pipelines._cli import build_parser, start_run
from creditwatch.pipelines.analyze import run_analyze
from creditwatch.pipelines.dashboard import run_dashboard
from creditwatch.pipelines.ingest import run_ingest
from creditwatch.pipelines.monitor import EVIDENTLY_MODES, run_monitor
from creditwatch.pipelines.simulate import run_simulate
from creditwatch.pipelines.train import run_train

logger = logging.getLogger("creditwatch.pipelines.run_all")

RUN_LOG_FILENAME = "pipeline_log.jsonl"
TRACKING_SUMMARY_FILENAME = "mlflow_summary.json"


def export_run_log(settings: Settings, run_id: str) -> Path | None:
    """Copia para ``reports/observability`` as linhas do log centralizado que pertencem a ``run_id``."""
    source = settings.logs_dir / LOG_FILENAME
    if not source.exists():
        return None
    for handler in logging.getLogger().handlers:
        handler.flush()
    lines = [
        line
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("run_id") == run_id
    ]
    target = settings.observability_dir / RUN_LOG_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")
    return target


def run_all(settings: Settings, evidently: str = "checkpoints") -> dict:
    """Executa todos os estagios em sequencia e devolve o resumo do monitoramento."""
    if settings.events_path.exists():
        settings.events_path.unlink()  # cada execucao completa comeca com o historico de eventos limpo
    run_ingest(settings, StageTracker("ingest", settings.events_path))
    run_train(settings, StageTracker("train", settings.events_path))
    run_simulate(settings, StageTracker("simulate", settings.events_path))
    overview = run_monitor(settings, StageTracker("monitor", settings.events_path), evidently=evidently)
    run_analyze(settings, StageTracker("analyze", settings.events_path))
    dashboard = run_dashboard(settings, StageTracker("dashboard", settings.events_path))
    export_tracking_summary(
        settings.mlflow_tracking_uri,
        (settings.mlflow_experiment_training, settings.mlflow_experiment_monitoring),
        settings.observability_dir / TRACKING_SUMMARY_FILENAME,
    )
    logger.info("pipeline completo concluido; dashboard em %s", dashboard)
    export_run_log(settings, current_run_id())
    return overview


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    parser = build_parser("Executa o pipeline completo de sustentacao do modelo de credito.")
    parser.add_argument("--evidently", choices=EVIDENTLY_MODES, default="checkpoints")
    args = parser.parse_args(argv)
    start_run(settings, args.log_level)
    run_all(settings, evidently=args.evidently)


if __name__ == "__main__":
    main()
