"""Logging centralizado: texto legivel no console e JSON estruturado em arquivo.

Todos os estagios do pipeline escrevem no mesmo arquivo ``logs/pipeline.jsonl``; cada linha
carrega o ``run_id`` da execucao, o que permite reconstruir uma execucao inteira com um filtro.

Privacidade: os logs registram contagens, nomes de colunas e de regras, nunca valores de linhas.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}
_run_id: ContextVar[str] = ContextVar("creditwatch_run_id", default="-")

LOG_FILENAME = "pipeline.jsonl"


def new_run_id() -> str:
    """Gera e ativa um identificador para a execucao corrente."""
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
    _run_id.set(run_id)
    return run_id


def current_run_id() -> str:
    return _run_id.get()


def set_run_id(run_id: str) -> None:
    _run_id.set(run_id)


class _RunIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = current_run_id()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO", logs_dir: Path | None = None) -> None:
    """Configura o logger raiz. Com ``logs_dir`` tambem grava JSON Lines em arquivo."""
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handlers: list[logging.Handler] = [console]
    if logs_dir is not None:
        logs_dir = Path(logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(logs_dir / LOG_FILENAME, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        handlers.append(file_handler)
    root = logging.getLogger()
    for handler in root.handlers[:]:
        handler.close()
    root.handlers[:] = handlers
    for handler in handlers:
        handler.addFilter(_RunIdFilter())
    root.setLevel(level.upper())
    # Bibliotecas verbosas nao devem poluir o log do pipeline.
    for noisy in ("mlflow", "alembic", "matplotlib", "urllib3", "evidently", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
