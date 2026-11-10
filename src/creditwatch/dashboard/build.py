"""Montagem do dashboard estatico de monitoramento (um unico HTML, sem dependencias externas).

O dashboard nao calcula nada: ele apenas le os artefatos gravados pelos estagios anteriores
(``reports/monitoring``, ``reports/observability``, ``reports/analysis``) e os embute como JSON em
um template. Os graficos sao SVG desenhados no navegador, entao o arquivo abre offline, em
qualquer maquina, e pode ser publicado como pagina estatica.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from creditwatch import __version__
from creditwatch.config.settings import Settings
from creditwatch.contracts.rules import RULES
from creditwatch.observability.alerts import read_alerts
from creditwatch.observability.events import read_events

TEMPLATE_DIR = Path(__file__).parent
TEMPLATE_NAME = "template.html.j2"
PHASE_LABELS = {
    "referencia": "Referência",
    "estavel": "Estável",
    "drift_de_dados": "Drift de dados",
    "drift_de_conceito": "Drift de conceito",
}
TIMELINE_FIELDS = (
    "roc_auc",
    "ks",
    "default_rate",
    "mean_score",
    "bad_rate_approved",
    "approval_rate",
    "calibration_gap",
    "n_drifted",
    "n_critical",
    "share_drifted",
    "score_psi",
    "alerts_critical",
    "alerts_warning",
    "latency_ms_per_1k_rows",
)


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _clean(value):
    """NaN do pandas vira ``None`` (JSON valido)."""
    if isinstance(value, float) and value != value:
        return None
    return value


def _phases(months: list[str], phase_by_month: dict[str, str]) -> list[dict]:
    """Intervalos contiguos de meses por fase, para as faixas dos graficos."""
    phases: list[dict] = []
    for index, month in enumerate(months):
        phase = phase_by_month[month]
        if phases and phases[-1]["phase"] == phase:
            phases[-1]["end"] = index
        else:
            phases.append({"phase": phase, "label": PHASE_LABELS.get(phase, phase), "start": index, "end": index})
    return phases


def _stage_stats(events: list[dict]) -> list[dict]:
    """Saude operacional por estagio: execucoes, resultados e duracao."""
    by_stage: dict[str, list[dict]] = {}
    for event in events:
        by_stage.setdefault(event["stage"], []).append(event)
    stats = []
    for stage, items in by_stage.items():
        durations = [item["duration_ms"] for item in items]
        stats.append(
            {
                "stage": stage,
                "runs": len(items),
                "success": sum(item["status"] == "success" for item in items),
                "blocked": sum(item["status"] == "blocked" for item in items),
                "failed": sum(item["status"] == "failed" for item in items),
                "median_ms": round(statistics.median(durations), 1),
                "cpu_ms": round(sum(item.get("cpu_ms") or 0 for item in items), 1),
                "max_ms": round(max(durations), 1),
                "rows": sum(item.get("rows_out") or 0 for item in items),
            }
        )
    return stats


def _latest_run(events: list[dict], pipeline: str) -> list[dict]:
    """Eventos da execucao mais recente de ``pipeline``."""
    scoped = [event for event in events if event["pipeline"] == pipeline]
    if not scoped:
        return []
    last_run = scoped[-1]["run_id"]
    return [event for event in scoped if event["run_id"] == last_run]


def build_payload(settings: Settings) -> dict:
    """Le os artefatos do monitoramento e devolve o dicionario embutido no dashboard."""
    reports = settings.reports_dir
    monitoring = reports / "monitoring"
    overview = _read_json(monitoring / "summary.json")
    if overview is None:
        raise FileNotFoundError(f"{monitoring / 'summary.json'} nao existe: rode o estagio monitor antes")

    timeline = pd.read_csv(monitoring / "timeline.csv")
    scored = timeline[timeline["status"] == "scored"].reset_index(drop=True)
    if scored.empty:
        raise ValueError("nenhuma remessa escorada: nao ha o que mostrar no dashboard")
    months = scored["month"].tolist()
    if len(set(months)) != len(months):
        raise ValueError("ha mais de uma remessa escorada no mesmo mes; o dashboard espera uma por mes")
    # Sem rotulo maduro em nenhuma remessa, as colunas de performance nem existem na serie temporal.
    scored = scored.reindex(columns=[*scored.columns, *(f for f in TIMELINE_FIELDS if f not in scored.columns)])
    batches = {path.stem: _read_json(path) for path in sorted((monitoring / "batches").glob("*.json"))}
    events = _latest_run(read_events(settings.events_path), "monitor")

    durations: dict[str, dict[str, float]] = {}
    for event in events:
        if event.get("batch"):
            durations.setdefault(event["batch"], {})[event["stage"]] = event["duration_ms"]

    deliveries = []
    for row in timeline.to_dict("records"):
        batch = batches[row["delivery"]]
        stages = durations.get(row["delivery"], {})
        deliveries.append(
            {
                "delivery": row["delivery"],
                "month": row["month"],
                "phase": PHASE_LABELS.get(row["phase"], row["phase"]),
                "status": row["status"],
                "note": batch.get("note", ""),
                "rows_received": int(row["rows_received"]),
                "rows_scored": int(row["rows_scored"]),
                "alerts_critical": int(row["alerts_critical"]),
                "alerts_warning": int(row["alerts_warning"]),
                "duration_ms": round(sum(stages.values()), 1),
                "stages": stages,
                "violations": batch["contract"].get("violations", []),
                "evidently": (batch.get("evidently") or {}).get("html"),
            }
        )

    scored_batches = [batches[delivery] for delivery in scored["delivery"]]
    feature_names = [feature["feature"] for feature in scored_batches[-1]["drift"]["features"]]
    psi_by_feature = {name: [] for name in feature_names}
    detail: dict[str, list[dict]] = {}
    for month, batch in zip(months, scored_batches, strict=True):
        by_name = {feature["feature"]: feature for feature in batch["drift"]["features"]}
        for name in feature_names:
            psi_by_feature[name].append(by_name[name]["psi"])
        detail[month] = batch["drift"]["features"]

    model = _read_json(settings.models_dir / "metadata.json") or {}
    analysis_dir = reports / "analysis"
    causal = _read_json(analysis_dir / "causal_attribution.json")
    diagnosis = _read_json(analysis_dir / "segment_diagnosis.json")
    fairness = _read_json(analysis_dir / "fairness_audit.json")
    remediation = _read_json(analysis_dir / "remediation.json")

    return {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "package_version": __version__,
        "model": {
            "version": model.get("model_version"),
            "type": model.get("model_type"),
            "threshold": model.get("threshold"),
            "trained_at": model.get("trained_at"),
            "importance": model.get("feature_importance", {}),
            "excluded": model.get("excluded_protected_attributes", []),
        },
        "reference": overview["reference_metrics"],
        "overview": overview,
        "thresholds": {
            "psi_warning": settings.psi_warning,
            "psi_critical": settings.psi_critical,
            "drift_share": settings.drift_share_alert,
            "auc_drop_warning": settings.auc_drop_warning,
            "auc_drop_critical": settings.auc_drop_critical,
            "calibration_gap": settings.calibration_gap_alert,
            "label_lag_months": settings.label_lag_months,
        },
        "phase_labels": PHASE_LABELS,
        "months": months,
        "phases": _phases(months, dict(zip(scored["month"], scored["phase"], strict=True))),
        "timeline": {field: [_clean(value) for value in scored[field].tolist()] for field in TIMELINE_FIELDS},
        "deliveries": deliveries,
        "features": {"names": feature_names, "psi": psi_by_feature, "detail": detail},
        "alerts": read_alerts(settings.alerts_path),
        "rules": [rule.as_dict() for rule in RULES],
        "raw_validation": _read_json(reports / "data_quality" / "raw_validation.json"),
        "stage_stats": _stage_stats(events),
        "pipeline": overview["pipeline"],
        "analysis": {
            "causal": causal,
            "segments": (diagnosis or {}).get("segments"),
            "utilization_cut": (diagnosis or {}).get("utilization_cut"),
            "scan": ((diagnosis or {}).get("calibration_scan") or {}).get("by_feature", [])[:6],
            "ranking": (diagnosis or {}).get("root_cause_ranking"),
            "fairness": (fairness or {}).get("worst_groups"),
            "remediation": remediation,
        },
    }


def render_dashboard(payload: dict) -> str:
    environment = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR), autoescape=select_autoescape(["html", "j2"]), trim_blocks=True
    )
    return environment.get_template(TEMPLATE_NAME).render(payload=payload)


def build_dashboard(settings: Settings, output: Path | None = None) -> Path:
    """Gera ``reports/dashboard/index.html`` e devolve o caminho."""
    output = Path(output) if output else settings.reports_dir / "dashboard" / "index.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_dashboard(build_payload(settings)), encoding="utf-8", newline="\n")
    return output
