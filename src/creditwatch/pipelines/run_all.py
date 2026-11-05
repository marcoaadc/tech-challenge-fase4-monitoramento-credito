"""Pipeline completo: ingestao -> treino -> simulacao -> monitoramento -> analises -> dashboard.

Todos os estagios compartilham o mesmo ``run_id`` e o mesmo arquivo de eventos, de modo que uma
execucao inteira pode ser reconstruida a partir de ``reports/observability/pipeline_events.jsonl``.
"""

from __future__ import annotations

import logging

from creditwatch.config.settings import Settings, get_settings
from creditwatch.observability.events import StageTracker
from creditwatch.pipelines._cli import build_parser, start_run
from creditwatch.pipelines.analyze import run_analyze
from creditwatch.pipelines.dashboard import run_dashboard
from creditwatch.pipelines.ingest import run_ingest
from creditwatch.pipelines.monitor import EVIDENTLY_MODES, run_monitor
from creditwatch.pipelines.simulate import run_simulate
from creditwatch.pipelines.train import run_train

logger = logging.getLogger("creditwatch.pipelines.run_all")


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
    logger.info("pipeline completo concluido; dashboard em %s", dashboard)
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
