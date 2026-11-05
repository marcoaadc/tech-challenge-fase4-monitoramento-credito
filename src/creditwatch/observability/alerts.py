"""Regras de alerta: transformam os resultados de um lote em alertas acionaveis.

Familias de alerta (``kind``):

- ``data_quality``: lote bloqueado pelo contrato de dados;
- ``data_drift``: drift nas features de entrada (disponivel no dia da escoragem);
- ``prediction_drift``: a distribuicao do score mudou (disponivel no dia da escoragem);
- ``target_drift``: a taxa de inadimplencia mudou (so apos a maturacao do rotulo);
- ``performance``: queda de AUC em relacao a referencia (so apos a maturacao do rotulo);
- ``calibration``: inadimplencia observada distante da PD prevista (so apos a maturacao).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from creditwatch.drift.detector import CRITICAL, WARNING, DriftReport

DATA_QUALITY = "data_quality"
DATA_DRIFT = "data_drift"
PREDICTION_DRIFT = "prediction_drift"
TARGET_DRIFT = "target_drift"
PERFORMANCE = "performance"
CALIBRATION = "calibration"
LABEL_DEPENDENT = (TARGET_DRIFT, PERFORMANCE, CALIBRATION)


@dataclass(frozen=True)
class Alert:
    batch: str
    kind: str
    severity: str  # "warning" | "critical"
    title: str
    detail: str
    metric: str
    value: float
    threshold: float
    detectable_at: str  # mes em que o alerta poderia disparar em producao

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AlertThresholds:
    psi_warning: float = 0.10
    psi_critical: float = 0.25
    auc_drop_warning: float = 0.03
    auc_drop_critical: float = 0.05
    calibration_gap: float = 0.05
    alpha: float = 0.05


def add_months(month: str, months: int) -> str:
    """Soma meses a uma string ``AAAA-MM``."""
    year, number = (int(part) for part in month.split("-"))
    index = year * 12 + (number - 1) + months
    return f"{index // 12}-{index % 12 + 1:02d}"


def data_quality_alert(batch: str, delivery: str, violated_rules: list[str], n_rows: int) -> Alert:
    return Alert(
        batch=batch,
        kind=DATA_QUALITY,
        severity=CRITICAL,
        title="Lote bloqueado pelo contrato de dados",
        detail=(
            f"Remessa {delivery}: regras violadas {', '.join(violated_rules)}. "
            f"{n_rows} linhas em quarentena; nada foi escorado."
        ),
        metric="contract_violated_rules",
        value=float(len(violated_rules)),
        threshold=0.0,
        detectable_at=batch,
    )


def drift_alerts(batch: str, report: DriftReport) -> list[Alert]:
    """Alerta de drift de entrada: no maximo um por lote, para nao gerar fadiga de alertas.

    O detalhe lista as features mais afetadas (maior PSI primeiro), que e por onde a
    investigacao de causa raiz deve comecar.
    """
    if not report.drifted_features:
        return []
    ranked = sorted((f for f in report.features if f.drifted), key=lambda f: -f.psi)
    top = ", ".join(f"{f.feature} (PSI {f.psi:.2f})" for f in ranked[:5])
    if report.dataset_drift:
        return [
            Alert(
                batch=batch,
                kind=DATA_DRIFT,
                severity=CRITICAL,
                title="Drift do dataset",
                detail=(
                    f"{len(ranked)} de {len(report.features)} features com drift ({report.share_drifted:.0%}; "
                    f"limite {report.thresholds.drift_share:.0%}). Mais afetadas: {top}."
                ),
                metric="share_drifted_features",
                value=round(report.share_drifted, 4),
                threshold=report.thresholds.drift_share,
                detectable_at=batch,
            )
        ]
    if report.critical_features:
        return [
            Alert(
                batch=batch,
                kind=DATA_DRIFT,
                severity=CRITICAL,
                title=f"Drift crítico em {len(report.critical_features)} feature(s)",
                detail=f"PSI >= {report.thresholds.psi_critical}. Mais afetadas: {top}.",
                metric="max_feature_psi",
                value=ranked[0].psi,
                threshold=report.thresholds.psi_critical,
                detectable_at=batch,
            )
        ]
    return [
        Alert(
            batch=batch,
            kind=DATA_DRIFT,
            severity=WARNING,
            title=f"Drift moderado em {len(ranked)} feature(s)",
            detail=f"Em observação: {top}.",
            metric="max_feature_psi",
            value=ranked[0].psi,
            threshold=report.thresholds.psi_warning,
            detectable_at=batch,
        )
    ]


def prediction_drift_alert(batch: str, score_psi: float, thresholds: AlertThresholds) -> Alert | None:
    if score_psi < thresholds.psi_warning:
        return None
    severity = CRITICAL if score_psi >= thresholds.psi_critical else WARNING
    limit = thresholds.psi_critical if severity == CRITICAL else thresholds.psi_warning
    return Alert(
        batch=batch,
        kind=PREDICTION_DRIFT,
        severity=severity,
        title="Drift na distribuição do score",
        detail=f"PSI do score {score_psi:.3f} (>= {limit}): o perfil de risco previsto mudou.",
        metric="psi_score",
        value=round(score_psi, 5),
        threshold=limit,
        detectable_at=batch,
    )


def performance_alerts(
    batch: str,
    metrics: dict[str, float],
    reference_metrics: dict[str, float],
    target_p_value: float,
    thresholds: AlertThresholds,
    label_lag_months: int,
) -> list[Alert]:
    """Alertas que dependem do rotulo: so podem disparar ``label_lag_months`` depois do lote."""
    detectable_at = add_months(batch, label_lag_months)
    alerts: list[Alert] = []

    auc_drop = reference_metrics["roc_auc"] - metrics["roc_auc"]
    if auc_drop >= thresholds.auc_drop_warning:
        severity = CRITICAL if auc_drop >= thresholds.auc_drop_critical else WARNING
        limit = thresholds.auc_drop_critical if severity == CRITICAL else thresholds.auc_drop_warning
        alerts.append(
            Alert(
                batch=batch,
                kind=PERFORMANCE,
                severity=severity,
                title="Queda de discriminação do modelo",
                detail=(
                    f"ROC AUC {metrics['roc_auc']:.3f} contra {reference_metrics['roc_auc']:.3f} na referência "
                    f"(queda de {auc_drop:.3f}); KS {metrics['ks']:.3f}."
                ),
                metric="roc_auc_drop",
                value=round(auc_drop, 5),
                threshold=limit,
                detectable_at=detectable_at,
            )
        )

    gap = abs(metrics["calibration_gap"])
    if gap >= thresholds.calibration_gap:
        alerts.append(
            Alert(
                batch=batch,
                kind=CALIBRATION,
                severity=CRITICAL if gap >= 2 * thresholds.calibration_gap else WARNING,
                title="Modelo descalibrado",
                detail=(
                    f"Inadimplência observada {metrics['default_rate']:.1%} contra PD média prevista "
                    f"{metrics['mean_score']:.1%} (diferença de {gap:.1%})."
                ),
                metric="calibration_gap",
                value=round(gap, 5),
                threshold=thresholds.calibration_gap,
                detectable_at=detectable_at,
            )
        )

    rate_change = metrics["default_rate"] - reference_metrics["default_rate"]
    if target_p_value < thresholds.alpha and abs(rate_change) >= thresholds.calibration_gap:
        alerts.append(
            Alert(
                batch=batch,
                kind=TARGET_DRIFT,
                severity=CRITICAL if abs(rate_change) >= 2 * thresholds.calibration_gap else WARNING,
                title="Mudança na taxa de inadimplência",
                detail=(
                    f"Taxa de inadimplência {metrics['default_rate']:.1%} contra "
                    f"{reference_metrics['default_rate']:.1%} na referência (p = {target_p_value:.1e})."
                ),
                metric="default_rate_change",
                value=round(rate_change, 5),
                threshold=thresholds.calibration_gap,
                detectable_at=detectable_at,
            )
        )
    return alerts


def write_alerts(alerts: list[Alert], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(alert.as_dict(), ensure_ascii=False) for alert in alerts]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def read_alerts(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
