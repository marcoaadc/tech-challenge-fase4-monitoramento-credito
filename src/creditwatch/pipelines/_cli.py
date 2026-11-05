"""Utilitarios compartilhados pelas CLIs dos estagios."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from creditwatch.config.settings import Settings
from creditwatch.observability.logs import configure_logging, new_run_id


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--log-level", default=None, help="Sobrescreve CREDITWATCH_LOG_LEVEL.")
    return parser


def start_run(settings: Settings, log_level: str | None = None) -> str:
    """Configura o logging centralizado e abre uma nova execucao (``run_id``)."""
    configure_logging(log_level or settings.log_level, settings.logs_dir)
    return new_run_id()


def _json_safe(value: Any) -> Any:
    """Converte NaN/inf em ``null`` e tipos numpy em tipos nativos (JSON valido)."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict | list) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_table(frame: pd.DataFrame, path: Path) -> Path:
    """Grava ``frame`` em Parquet ou CSV, conforme a extensao."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        frame.to_csv(path, index=False, lineterminator="\n")
    else:
        frame.to_parquet(path, index=False)
    return path


def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
