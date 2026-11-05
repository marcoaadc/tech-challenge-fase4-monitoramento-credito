"""Estagio 6 - dashboard: consolida os artefatos do monitoramento em um HTML estatico."""

from __future__ import annotations

import logging
from pathlib import Path

from creditwatch.config.settings import Settings, get_settings
from creditwatch.dashboard.build import build_dashboard
from creditwatch.observability.events import StageTracker
from creditwatch.pipelines._cli import build_parser, start_run

logger = logging.getLogger("creditwatch.pipelines.dashboard")


def run_dashboard(settings: Settings, tracker: StageTracker | None = None, output: Path | None = None) -> Path:
    tracker = tracker or StageTracker("dashboard", settings.events_path)
    with tracker.stage("render_dashboard") as event:
        path = build_dashboard(settings, output)
        event.details = {"path": path.as_posix(), "size_kb": round(path.stat().st_size / 1024, 1)}
    logger.info("dashboard gravado em %s", path)
    return path


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    parser = build_parser("Gera o dashboard estatico de monitoramento.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    start_run(settings, args.log_level)
    run_dashboard(settings, output=args.output)


if __name__ == "__main__":
    main()
