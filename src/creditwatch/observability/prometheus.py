"""Exportacao das metricas de saude em formato Prometheus (integracao com a stack da Fase 3).

O pipeline e um job em lote, entao nao ha endpoint ``/metrics`` para ser raspado. As metricas sao
gravadas em um arquivo de exposicao (``metrics.prom``, lido pelo *textfile collector* do
node_exporter) e, se ``CREDITWATCH_PUSHGATEWAY_URL`` estiver definida, enviadas a um Pushgateway.
As series refletem o estado do ultimo lote processado; o historico fica no MLflow e no dashboard.
"""

from __future__ import annotations

import logging
from pathlib import Path

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    disable_created_metrics,
    push_to_gateway,
    write_to_textfile,
)

from creditwatch.observability.events import StageEvent

logger = logging.getLogger("creditwatch.observability.prometheus")

# As series *_created so fazem sentido em processos de longa duracao; em job de lote sao ruido.
disable_created_metrics()

JOB_NAME = "creditwatch_monitoring"
DURATION_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)


class PipelineMetrics:
    """Registro isolado (um por execucao) com as metricas operacionais e de modelo."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.stage_runs = Counter(
            "creditwatch_stage_runs_total",
            "Execucoes de estagios do pipeline por resultado.",
            ["stage", "status"],
            registry=r,
        )
        self.stage_duration = Histogram(
            "creditwatch_stage_duration_seconds",
            "Duracao dos estagios do pipeline.",
            ["stage"],
            buckets=DURATION_BUCKETS,
            registry=r,
        )
        self.rows = Counter("creditwatch_rows_total", "Linhas processadas por destino.", ["outcome"], registry=r)
        self.batches = Counter("creditwatch_batches_total", "Lotes processados por resultado.", ["status"], registry=r)
        self.contract_violations = Counter(
            "creditwatch_contract_violations_total", "Violacoes do contrato de dados por regra.", ["rule"], registry=r
        )
        self.alerts = Counter("creditwatch_alerts_total", "Alertas disparados.", ["kind", "severity"], registry=r)
        self.feature_psi = Gauge(
            "creditwatch_feature_psi",
            "PSI de cada feature no ultimo lote, contra a referencia.",
            ["feature"],
            registry=r,
        )
        self.drifted_features = Gauge(
            "creditwatch_drifted_features", "Numero de features com drift no ultimo lote.", registry=r
        )
        self.drift_share = Gauge("creditwatch_drift_share", "Fracao de features com drift no ultimo lote.", registry=r)
        self.score_psi = Gauge("creditwatch_score_psi", "PSI do score (drift de predicao) no ultimo lote.", registry=r)
        self.approval_rate = Gauge("creditwatch_approval_rate", "Taxa de aprovacao no ultimo lote.", registry=r)
        self.model_auc = Gauge("creditwatch_model_roc_auc", "ROC AUC no ultimo lote com rotulo maduro.", registry=r)
        self.model_ks = Gauge("creditwatch_model_ks", "KS do modelo no ultimo lote com rotulo maduro.", registry=r)
        self.default_rate = Gauge(
            "creditwatch_default_rate", "Taxa de inadimplencia no ultimo lote com rotulo maduro.", registry=r
        )
        self.bad_rate_approved = Gauge(
            "creditwatch_bad_rate_approved", "Inadimplencia entre os aprovados no ultimo lote com rotulo.", registry=r
        )
        self.last_batch = Gauge(
            "creditwatch_last_batch_info",
            "Ultimo lote processado (valor constante 1).",
            ["batch", "model_version"],
            registry=r,
        )
        self.last_run_timestamp = Gauge(
            "creditwatch_last_run_timestamp_seconds", "Instante do termino da ultima execucao.", registry=r
        )

    # --- Atualizacao ---------------------------------------------------------
    def observe_stage(self, event: StageEvent) -> None:
        self.stage_runs.labels(stage=event.stage, status=event.status).inc()
        self.stage_duration.labels(stage=event.stage).observe(event.duration_ms / 1000)

    def observe_batch(self, summary: dict, model_version: str) -> None:
        """Atualiza as series a partir do resumo de um lote (ver ``pipelines.monitor``)."""
        status = summary["status"]
        self.batches.labels(status=status).inc()
        if status == "blocked":
            self.rows.labels(outcome="quarantined").inc(summary["rows_received"])
            for rule in summary["contract"]["violated_rules"]:
                self.contract_violations.labels(rule=rule).inc()
        else:
            self.rows.labels(outcome="scored").inc(summary["rows_scored"])
            drift = summary["drift"]
            for feature in drift["features"]:
                self.feature_psi.labels(feature=feature["feature"]).set(feature["psi"])
            self.drifted_features.set(drift["n_drifted"])
            self.drift_share.set(drift["share_drifted"])
            self.score_psi.set(summary["prediction"]["score_psi"])
            self.approval_rate.set(summary["prediction"]["approval_rate"])
            performance = summary.get("performance")
            if performance:
                self.model_auc.set(performance["roc_auc"])
                self.model_ks.set(performance["ks"])
                self.default_rate.set(performance["default_rate"])
                self.bad_rate_approved.set(performance["bad_rate_approved"])
            self.last_batch.clear()
            self.last_batch.labels(batch=summary["batch"], model_version=model_version).set(1)
        for alert in summary["alerts"]:
            self.alerts.labels(kind=alert["kind"], severity=alert["severity"]).inc()

    # --- Exportacao ----------------------------------------------------------
    def export(self, path: Path, pushgateway_url: str | None = None) -> Path:
        """Grava o arquivo de exposicao e, opcionalmente, envia ao Pushgateway."""
        self.last_run_timestamp.set_to_current_time()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_to_textfile(str(path), self.registry)
        if pushgateway_url:
            try:
                push_to_gateway(pushgateway_url, job=JOB_NAME, registry=self.registry)
                logger.info("metricas enviadas ao Pushgateway %s", pushgateway_url)
            except OSError as exc:  # falha de observabilidade nao derruba o pipeline
                logger.warning("falha ao enviar metricas ao Pushgateway: %s", exc)
        return path
