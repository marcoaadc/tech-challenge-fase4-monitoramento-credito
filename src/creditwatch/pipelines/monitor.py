"""Estagio 4 - monitoramento: processa cada remessa de producao de ponta a ponta.

Para cada remessa, na ordem do manifesto:

1. ``read_batch``    le o arquivo recebido;
2. ``contract``      aplica o contrato de dados; se reprovar, a remessa vai para a quarentena, um
                     alerta e emitido e **nada e escorado**;
3. ``score``         calcula a PD e a decisao de credito;
4. ``drift``         compara features e score com o Dataset de Referencia (PSI, KS, qui-quadrado);
5. ``performance``   quando o rotulo amadurece, mede AUC, KS, calibracao e taxa de inadimplencia;
6. ``evidently``     gera o relatorio HTML do Evidently nos meses de checkpoint.

Os resultados alimentam os alertas, o MLflow, o arquivo de metricas Prometheus e o dashboard.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import pandas as pd

from creditwatch.config.settings import Settings, get_settings
from creditwatch.contracts.credit_contract import build_contract, build_label_contract
from creditwatch.contracts.gate import DataContractError, enforce_contract, write_validation_report
from creditwatch.data.schema import DECISION, ID_COLUMN, SCORE, TARGET
from creditwatch.data.split import SPLIT_COLUMN
from creditwatch.drift import evidently_report
from creditwatch.drift.detector import DriftThresholds, detect_drift
from creditwatch.drift.statistics import ks_test, psi_numerical, two_proportion_test
from creditwatch.features.engineering import add_engineered_features
from creditwatch.model.metrics import classification_metrics, score_summary
from creditwatch.model.registry import CreditScorer
from creditwatch.observability.alerts import (
    Alert,
    AlertThresholds,
    add_months,
    data_quality_alert,
    drift_alerts,
    label_quality_alert,
    performance_alerts,
    prediction_drift_alert,
    write_alerts,
)
from creditwatch.observability.events import StageBlocked, StageTracker
from creditwatch.observability.prometheus import PipelineMetrics
from creditwatch.observability.tracking import Tracker
from creditwatch.pipelines._cli import build_parser, read_json, read_table, start_run, write_json, write_table
from creditwatch.pipelines.simulate import MANIFEST_FILENAME

logger = logging.getLogger("creditwatch.pipelines.monitor")

SCORED, BLOCKED = "scored", "blocked"
EVIDENTLY_MODES = ("checkpoints", "all", "none")


@dataclass
class ReferenceBundle:
    """Linhas de base do monitoramento, todas derivadas do Dataset de Referencia."""

    full: pd.DataFrame  # referencia completa com features de engenharia: base do drift de entrada
    holdout: pd.DataFrame  # particoes val+test (fora do ajuste) com score: base da distribuicao do score
    metrics: dict[str, float]  # metricas do modelo na particao de teste: base de performance

    @classmethod
    def build(cls, reference: pd.DataFrame, scorer: CreditScorer) -> ReferenceBundle:
        """Monta as linhas de base.

        A performance de referencia usa so a particao ``test``: a ``val`` escolheu o modelo e o
        ponto de corte, e medir nela daria uma base otimista (alertas de queda disparariam cedo).
        Para comparar distribuicoes de score, ``val`` e ``test`` juntas dao uma base maior e
        continuam fora do ajuste do modelo.
        """
        full = add_engineered_features(reference)
        holdout = full[full[SPLIT_COLUMN] != "train"].copy()
        holdout[SCORE] = scorer.score(holdout)
        test = holdout[holdout[SPLIT_COLUMN] == "test"]
        metrics = classification_metrics(test[TARGET].to_numpy(), test[SCORE].to_numpy(), scorer.threshold)
        return cls(full=full, holdout=holdout, metrics=metrics)


def checkpoint_months(deliveries: list[dict]) -> set[str]:
    """Ultimo mes de cada fase do cenario: onde o relatorio do Evidently e gerado por padrao."""
    last_by_phase: dict[str, str] = {}
    for delivery in deliveries:
        last_by_phase[delivery["phase"]] = max(delivery["month"], last_by_phase.get(delivery["phase"], ""))
    return set(last_by_phase.values())


def _thresholds(settings: Settings) -> tuple[DriftThresholds, AlertThresholds]:
    drift = DriftThresholds(
        psi_warning=settings.psi_warning,
        psi_critical=settings.psi_critical,
        effect_size=settings.ks_effect_size,
        alpha=settings.p_value_alpha,
        drift_share=settings.drift_share_alert,
    )
    alert = AlertThresholds(
        psi_warning=settings.psi_warning,
        psi_critical=settings.psi_critical,
        auc_drop_warning=settings.auc_drop_warning,
        auc_drop_critical=settings.auc_drop_critical,
        calibration_gap=settings.calibration_gap_alert,
        alpha=settings.p_value_alpha,
    )
    return drift, alert


def process_delivery(
    delivery: dict,
    settings: Settings,
    scorer: CreditScorer,
    reference: ReferenceBundle,
    tracker: StageTracker,
    with_evidently: bool,
) -> dict:
    """Processa uma remessa e devolve o seu resumo (ver docstring do modulo)."""
    batch_id, month = delivery["delivery"], delivery["month"]
    production_dir = settings.production_dir
    drift_thresholds, alert_thresholds = _thresholds(settings)
    alerts: list[Alert] = []
    summary: dict = {
        "delivery": batch_id,
        "batch": month,
        "phase": delivery["phase"],
        "note": delivery.get("note", ""),
        "plan": delivery.get("plan", {}),
        "status": SCORED,
        "rows_received": 0,
        "rows_scored": 0,
        "contract": {},
        "prediction": None,
        "drift": None,
        "evidently": None,
        "performance": None,
        "alerts": [],
    }

    with tracker.stage("read_batch", batch=batch_id) as event:
        raw_batch = read_table(production_dir / delivery["batch_file"])
        event.rows_out = summary["rows_received"] = len(raw_batch)

    with tracker.stage("contract", batch=batch_id, rows_in=len(raw_batch)) as event:
        try:
            batch, result = enforce_contract(
                raw_batch, build_contract(require_target=False), batch=batch_id, quarantine_dir=settings.quarantine_dir
            )
        except DataContractError as exc:
            result = exc.result
            write_validation_report(result, settings.reports_dir / "data_quality" / f"blocked_{batch_id}.json")
            summary["status"] = BLOCKED
            summary["contract"] = {
                "status": "bloqueado",
                "violated_rules": result.violated_rules,
                "rows_with_failures": result.rows_with_failures,
                "violations": [violation.as_dict() for violation in result.violations],
                "warnings": result.warnings,
            }
            alerts.append(data_quality_alert(month, batch_id, result.violated_rules, len(raw_batch)))
            summary["alerts"] = [alert.as_dict() for alert in alerts]
            event.rows_out = 0
            event.details = {"violated_rules": result.violated_rules}
            raise StageBlocked(str(exc)) from exc
        summary["contract"] = {"status": "aprovado", "violated_rules": [], "warnings": result.warnings}
        event.rows_out = len(batch)

    if summary["status"] == BLOCKED:
        return summary

    with tracker.stage("score", batch=batch_id, rows_in=len(batch)) as event:
        started = time.perf_counter()
        scored = add_engineered_features(batch)
        scored[SCORE] = scorer.score(scored)
        scored[DECISION] = scorer.decide(scored[SCORE].to_numpy())
        elapsed = time.perf_counter() - started
        predictions = scored[[ID_COLUMN, SCORE, DECISION]].assign(model_version=scorer.metadata.model_version)
        write_table(predictions, settings.data_dir / "predictions" / f"predictions_{month}.parquet")
        event.rows_out = summary["rows_scored"] = len(scored)
        latency = round(elapsed * 1000 / max(len(scored), 1) * 1000, 2)
        event.details = {"latency_ms_per_1k_rows": latency}

    with tracker.stage("drift", batch=batch_id, rows_in=len(scored)) as event:
        drift = detect_drift(reference.full, scored, drift_thresholds)
        score_test = ks_test(reference.holdout[SCORE], scored[SCORE])
        score_psi = psi_numerical(reference.holdout[SCORE], scored[SCORE])
        summary["drift"] = drift.as_dict()
        summary["prediction"] = {
            **score_summary(scored[SCORE].to_numpy(), scorer.threshold),
            "score_psi": round(score_psi, 5),
            "score_ks": round(score_test.statistic, 5),
            "score_ks_p_value": score_test.p_value,
            "latency_ms_per_1k_rows": latency,
        }
        alerts.extend(drift_alerts(month, drift))
        prediction_alert = prediction_drift_alert(month, score_psi, alert_thresholds)
        if prediction_alert:
            alerts.append(prediction_alert)
        event.details = {"n_drifted": len(drift.drifted_features), "severity": drift.severity}

    if delivery.get("labels_file"):
        # Rotulos invalidos nao derrubam o monitoramento: a performance da remessa fica indisponivel,
        # um alerta e emitido e as remessas seguintes continuam sendo processadas.
        with tracker.stage("performance", batch=batch_id, rows_in=len(scored)) as event:
            try:
                labels, _ = enforce_contract(
                    read_table(production_dir / delivery["labels_file"]),
                    build_label_contract(),
                    batch=f"labels-{batch_id}",
                )
                problem = None
            except DataContractError as exc:
                problem = f"regras violadas {', '.join(exc.result.violated_rules)}"
            if problem is None:
                labeled = scored.merge(labels, on=ID_COLUMN, how="inner", validate="one_to_one")
                missing = len(scored) - len(labeled)
                if missing:
                    problem = f"faltam rotulos de {missing} propostas escoradas"
            if problem is not None:
                alerts.append(label_quality_alert(month, batch_id, problem))
                summary["alerts"] = [alert.as_dict() for alert in alerts]
                summary["labels"] = {"status": "bloqueado", "problem": problem}
                event.rows_out = 0
                raise StageBlocked(f"rotulos de {batch_id} rejeitados: {problem}")
            scored = labeled
            metrics = classification_metrics(scored[TARGET].to_numpy(), scored[SCORE].to_numpy(), scorer.threshold)
            reference_n = int(reference.metrics["n"])
            target_test = two_proportion_test(
                round(reference.metrics["default_rate"] * reference_n),
                reference_n,
                int(scored[TARGET].sum()),
                len(scored),
            )
            summary["performance"] = {
                **metrics,
                "roc_auc_drop": reference.metrics["roc_auc"] - metrics["roc_auc"],
                "default_rate_change": target_test.statistic,
                "default_rate_p_value": target_test.p_value,
                "labels_available_at": add_months(month, settings.label_lag_months),
            }
            alerts.extend(
                performance_alerts(
                    month, metrics, reference.metrics, target_test.p_value, alert_thresholds, settings.label_lag_months
                )
            )
            event.rows_out = len(scored)
            event.details = {"roc_auc": round(metrics["roc_auc"], 4), "ks": round(metrics["ks"], 4)}

    if with_evidently:
        with tracker.stage("evidently", batch=batch_id, rows_in=len(scored)) as event:
            snapshot = evidently_report.run_evidently_report(
                reference.holdout,
                scored,
                psi_threshold=settings.psi_warning,
                drift_share=settings.drift_share_alert,
                decision_threshold=scorer.threshold if summary["performance"] else None,
                batch=month,
            )
            html_path = evidently_report.save_html(snapshot, settings.reports_dir / "evidently" / f"drift_{month}.html")
            summary["evidently"] = {
                **evidently_report.summarize(snapshot, settings.psi_warning),
                "html": html_path.relative_to(settings.reports_dir).as_posix(),
            }
            event.details = {"n_drifted": summary["evidently"]["n_drifted"]}

    summary["alerts"] = [alert.as_dict() for alert in alerts]
    return summary


def timeline_row(summary: dict) -> dict:
    """Uma linha da serie temporal consolidada (``reports/monitoring/timeline.csv``)."""
    alerts = summary["alerts"]
    row: dict = {
        "delivery": summary["delivery"],
        "month": summary["batch"],
        "phase": summary["phase"],
        "status": summary["status"],
        "rows_received": summary["rows_received"],
        "rows_scored": summary["rows_scored"],
        "alerts_critical": sum(alert["severity"] == "critical" for alert in alerts),
        "alerts_warning": sum(alert["severity"] == "warning" for alert in alerts),
        **{f"plan_{key}": value for key, value in summary["plan"].items() if key not in ("month", "phase")},
    }
    if summary["status"] == SCORED:
        drift, prediction = summary["drift"], summary["prediction"]
        row.update(
            n_drifted=drift["n_drifted"],
            n_critical=len(drift["critical_features"]),
            share_drifted=drift["share_drifted"],
            drift_severity=drift["severity"],
            score_psi=prediction["score_psi"],
            mean_score=prediction["mean_score"],
            approval_rate=prediction["approval_rate"],
            latency_ms_per_1k_rows=prediction["latency_ms_per_1k_rows"],
        )
        performance = summary["performance"]
        if performance:
            row.update(
                {
                    key: performance[key]
                    for key in (
                        "roc_auc",
                        "ks",
                        "gini",
                        "brier",
                        "precision",
                        "recall",
                        "default_rate",
                        "calibration_gap",
                        "bad_rate_approved",
                        "roc_auc_drop",
                        "labels_available_at",
                    )
                }
            )
    return row


def _mlflow_metrics(summary: dict) -> dict[str, float]:
    metrics: dict[str, float] = {
        "pipeline/rows_received": summary["rows_received"],
        "pipeline/rows_scored": summary["rows_scored"],
        "pipeline/blocked": float(summary["status"] == BLOCKED),
        "alerts/critical": sum(alert["severity"] == "critical" for alert in summary["alerts"]),
        "alerts/warning": sum(alert["severity"] == "warning" for alert in summary["alerts"]),
    }
    if summary["status"] == SCORED:
        metrics.update(
            {
                "drift/n_drifted": summary["drift"]["n_drifted"],
                "drift/share_drifted": summary["drift"]["share_drifted"],
                "prediction/score_psi": summary["prediction"]["score_psi"],
                "prediction/approval_rate": summary["prediction"]["approval_rate"],
                "prediction/mean_score": summary["prediction"]["mean_score"],
            }
        )
        metrics.update({f"psi/{f['feature']}": f["psi"] for f in summary["drift"]["features"]})
        if summary["performance"]:
            metrics.update({f"performance/{key}": value for key, value in summary["performance"].items()})
    return metrics


def run_monitor(settings: Settings, tracker: StageTracker | None = None, evidently: str = "checkpoints") -> dict:
    if evidently not in EVIDENTLY_MODES:
        raise ValueError(f"evidently deve ser um de {EVIDENTLY_MODES}")
    metrics = PipelineMetrics()
    tracker = tracker or StageTracker("monitor", settings.events_path)
    tracker.metrics = metrics
    mlflow_tracker = Tracker(settings.mlflow_tracking_uri, settings.mlflow_experiment_monitoring)
    monitoring_dir = settings.reports_dir / "monitoring"

    with tracker.stage("load_baselines") as event:
        manifest = read_json(settings.production_dir / MANIFEST_FILENAME)
        scorer = CreditScorer.load(settings.models_dir)
        reference = ReferenceBundle.build(read_table(settings.reference_path), scorer)
        event.rows_out = len(reference.full)

    deliveries = manifest["deliveries"]
    if evidently == "all":
        evidently_months = {delivery["month"] for delivery in deliveries}
    else:
        evidently_months = checkpoint_months(deliveries) if evidently == "checkpoints" else set()

    summaries: list[dict] = []
    with mlflow_tracker.run(f"monitoramento-{scorer.metadata.model_version}", tags={"stage": "monitoring"}):
        mlflow_tracker.log_params(
            {
                "model_version": scorer.metadata.model_version,
                "threshold": scorer.threshold,
                "n_deliveries": len(deliveries),
                "psi_warning": settings.psi_warning,
                "psi_critical": settings.psi_critical,
                "drift_share_alert": settings.drift_share_alert,
                "label_lag_months": settings.label_lag_months,
            }
        )
        mlflow_tracker.log_metrics({f"reference/{key}": value for key, value in reference.metrics.items()})
        for step, delivery in enumerate(deliveries):
            summary = process_delivery(
                delivery, settings, scorer, reference, tracker, with_evidently=delivery["month"] in evidently_months
            )
            summaries.append(summary)
            summary_path = write_json(monitoring_dir / "batches" / f"{summary['delivery']}.json", summary)
            metrics.observe_batch(summary, scorer.metadata.model_version)

            batch_metrics = _mlflow_metrics(summary)
            mlflow_tracker.log_metrics(batch_metrics, step=step)
            with mlflow_tracker.run(
                summary["delivery"],
                nested=True,
                tags={"status": summary["status"], "phase": summary["phase"], "month": summary["batch"]},
            ):
                mlflow_tracker.log_params(summary["plan"])
                mlflow_tracker.log_metrics(batch_metrics)
                mlflow_tracker.log_artifact(summary_path)
                if summary["evidently"]:
                    mlflow_tracker.log_artifact(settings.reports_dir / summary["evidently"]["html"], "evidently")

    timeline = pd.DataFrame([timeline_row(summary) for summary in summaries])
    timeline.to_csv(monitoring_dir / "timeline.csv", index=False, lineterminator="\n")
    feature_rows = [
        {"month": s["batch"], "phase": s["phase"], **feature}
        for s in summaries
        if s["status"] == SCORED
        for feature in s["drift"]["features"]
    ]
    pd.DataFrame(feature_rows).to_csv(monitoring_dir / "feature_drift.csv", index=False, lineterminator="\n")

    all_alerts = [Alert(**alert) for summary in summaries for alert in summary["alerts"]]
    write_alerts(all_alerts, settings.alerts_path)
    metrics.export(settings.observability_dir / "metrics.prom", settings.pushgateway_url)

    overview = {
        "model_version": scorer.metadata.model_version,
        "threshold": scorer.threshold,
        "reference_metrics": reference.metrics,
        "deliveries": len(summaries),
        "scored": sum(s["status"] == SCORED for s in summaries),
        "blocked": sum(s["status"] == BLOCKED for s in summaries),
        "rows_scored": int(sum(s["rows_scored"] for s in summaries)),
        "rows_quarantined": int(sum(s["rows_received"] for s in summaries if s["status"] == BLOCKED)),
        "alerts": {
            "total": len(all_alerts),
            "critical": sum(alert.severity == "critical" for alert in all_alerts),
            "warning": sum(alert.severity == "warning" for alert in all_alerts),
            "by_kind": pd.Series([alert.kind for alert in all_alerts]).value_counts().to_dict() if all_alerts else {},
        },
        "first_drift_alert": next((a.batch for a in all_alerts if a.kind == "data_drift"), None),
        "first_performance_alert": next((a.batch for a in all_alerts if a.kind == "performance"), None),
        "evidently_reports": sorted(s["evidently"]["html"] for s in summaries if s["evidently"]),
        "pipeline": tracker.summary(),
    }
    write_json(monitoring_dir / "summary.json", overview)
    logger.info(
        "monitoramento concluido: %d remessas (%d bloqueadas), %d alertas (%d criticos)",
        overview["deliveries"],
        overview["blocked"],
        overview["alerts"]["total"],
        overview["alerts"]["critical"],
    )
    return overview


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    parser = build_parser("Monitora as remessas de producao: contrato, escoragem, drift, performance e alertas.")
    parser.add_argument("--evidently", choices=EVIDENTLY_MODES, default="checkpoints")
    args = parser.parse_args(argv)
    start_run(settings, args.log_level)
    run_monitor(settings, evidently=args.evidently)


if __name__ == "__main__":
    main()
