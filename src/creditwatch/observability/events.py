"""Eventos de execucao dos estagios do pipeline (saude operacional).

Cada estagio roda dentro de ``tracker.stage(...)``, que mede a duracao, captura sucesso, falha ou
bloqueio e grava um evento estruturado em ``pipeline_events.jsonl``. Esse arquivo e a fonte unica
das metricas operacionais do dashboard, do MLflow e do Prometheus.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from creditwatch.observability.logs import current_run_id

logger = logging.getLogger("creditwatch.pipeline")

SUCCESS, FAILED, BLOCKED = "success", "failed", "blocked"


class StageBlocked(Exception):  # noqa: N818 - sinal de controle, nao um erro
    """Levantada dentro de um estagio para marca-lo como bloqueado (ex.: contrato reprovado)."""


@dataclass
class StageEvent:
    run_id: str
    pipeline: str
    stage: str
    batch: str | None
    status: str = SUCCESS
    started_at: str = ""
    duration_ms: float = 0.0
    cpu_ms: float = 0.0  # tempo de CPU do processo no estagio (uso de recurso, nao tempo de espera)
    rows_in: int | None = None
    rows_out: int | None = None
    error: str | None = None
    details: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


class StageTracker:
    """Registra os eventos de uma execucao e os persiste em JSON Lines."""

    def __init__(self, pipeline: str, events_path: Path | None = None, metrics=None):
        self.pipeline = pipeline
        self.events_path = Path(events_path) if events_path else None
        self.metrics = metrics
        self.events: list[StageEvent] = []
        if self.events_path:
            self.events_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def stage(self, name: str, batch: str | None = None, rows_in: int | None = None) -> Iterator[StageEvent]:
        """Executa um estagio medindo duracao e resultado.

        ``StageBlocked`` marca o estagio como bloqueado e e absorvida (o pipeline segue para o
        proximo lote); qualquer outra excecao marca falha e e propagada.
        """
        event = StageEvent(
            run_id=current_run_id(),
            pipeline=self.pipeline,
            stage=name,
            batch=batch,
            started_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            rows_in=rows_in,
        )
        started, cpu_started = time.perf_counter(), time.process_time()
        try:
            yield event
        except StageBlocked as blocked:
            event.status = BLOCKED
            event.error = str(blocked)
        except Exception as exc:
            event.status = FAILED
            event.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            event.duration_ms = round((time.perf_counter() - started) * 1000, 2)
            event.cpu_ms = round((time.process_time() - cpu_started) * 1000, 2)
            self._record(event)

    def _record(self, event: StageEvent) -> None:
        self.events.append(event)
        if self.events_path:
            with self.events_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(event.as_dict(), ensure_ascii=False, default=str) + "\n")
        if self.metrics is not None:
            self.metrics.observe_stage(event)
        log = logger.info if event.status == SUCCESS else logger.error
        log(
            "estagio %s%s: %s em %.0f ms",
            event.stage,
            f" [{event.batch}]" if event.batch else "",
            event.status,
            event.duration_ms,
            extra={
                "event": "stage_finished",
                "pipeline": event.pipeline,
                "stage": event.stage,
                "batch": event.batch,
                "status": event.status,
                "duration_ms": event.duration_ms,
                "cpu_ms": event.cpu_ms,
                "rows_in": event.rows_in,
                "rows_out": event.rows_out,
            },
        )

    def summary(self) -> dict:
        """Resumo de saude da execucao: contagem por status e duracao total."""
        by_status = dict.fromkeys((SUCCESS, FAILED, BLOCKED), 0)
        for event in self.events:
            by_status[event.status] += 1
        return {
            "run_id": current_run_id(),
            "pipeline": self.pipeline,
            "stages": len(self.events),
            **by_status,
            "duration_ms": round(sum(event.duration_ms for event in self.events), 2),
            "cpu_ms": round(sum(event.cpu_ms for event in self.events), 2),
        }


def read_events(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
