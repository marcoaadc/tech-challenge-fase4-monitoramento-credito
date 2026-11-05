"""Deteccao de drift por feature e consolidacao por lote.

Compara um lote de producao com o Dataset de Referencia e classifica cada feature em
``ok``, ``warning`` ou ``critical`` segundo limiares de tamanho de efeito (ver ``Settings``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from creditwatch.data.schema import CATEGORICAL_MONITORED, MODEL_FEATURES, NUMERICAL_MONITORED
from creditwatch.drift.statistics import chi_square_test, ks_test, psi_categorical, psi_numerical

OK, WARNING, CRITICAL = "ok", "warning", "critical"
SEVERITY_ORDER = {OK: 0, WARNING: 1, CRITICAL: 2}


@dataclass(frozen=True)
class DriftThresholds:
    psi_warning: float = 0.10
    psi_critical: float = 0.25
    effect_size: float = 0.10  # D do KS (numericas) ou V de Cramer (categoricas)
    alpha: float = 0.05
    drift_share: float = 0.30


@dataclass(frozen=True)
class FeatureDrift:
    feature: str
    kind: str  # "numerical" | "categorical"
    model_input: bool
    psi: float
    test: str  # "ks" | "chi2"
    statistic: float  # D do KS ou V de Cramer
    p_value: float
    reference_mean: float
    current_mean: float
    severity: str

    @property
    def drifted(self) -> bool:
        return self.severity != OK

    def as_dict(self) -> dict:
        return {**asdict(self), "drifted": self.drifted}


@dataclass
class DriftReport:
    features: list[FeatureDrift] = field(default_factory=list)
    thresholds: DriftThresholds = field(default_factory=DriftThresholds)

    @property
    def drifted_features(self) -> list[str]:
        return [f.feature for f in self.features if f.drifted]

    @property
    def critical_features(self) -> list[str]:
        return [f.feature for f in self.features if f.severity == CRITICAL]

    @property
    def share_drifted(self) -> float:
        return len(self.drifted_features) / len(self.features) if self.features else 0.0

    @property
    def dataset_drift(self) -> bool:
        return self.share_drifted >= self.thresholds.drift_share

    @property
    def severity(self) -> str:
        if self.dataset_drift or self.critical_features:
            return CRITICAL
        return WARNING if self.drifted_features else OK

    def as_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame([f.as_dict() for f in self.features])
        return frame.sort_values("psi", ascending=False).reset_index(drop=True)

    def as_dict(self) -> dict:
        return {
            "n_features": len(self.features),
            "n_drifted": len(self.drifted_features),
            "share_drifted": round(self.share_drifted, 4),
            "dataset_drift": self.dataset_drift,
            "severity": self.severity,
            "drifted_features": self.drifted_features,
            "critical_features": self.critical_features,
            "thresholds": asdict(self.thresholds),
            "features": [f.as_dict() for f in sorted(self.features, key=lambda f: -f.psi)],
        }


def classify(psi: float, statistic: float, p_value: float, thresholds: DriftThresholds) -> str:
    """Severidade de uma feature: o PSI manda; o teste de hipotese so eleva de ``ok`` para ``warning``."""
    if psi >= thresholds.psi_critical:
        return CRITICAL
    significant_effect = statistic >= thresholds.effect_size and p_value < thresholds.alpha
    if psi >= thresholds.psi_warning or significant_effect:
        return WARNING
    return OK


def detect_feature_drift(
    reference: pd.Series, current: pd.Series, kind: str, thresholds: DriftThresholds, model_input: bool = True
) -> FeatureDrift:
    if kind == "numerical":
        psi = psi_numerical(reference, current)
        result, test = ks_test(reference, current), "ks"
    else:
        psi = psi_categorical(reference, current)
        result, test = chi_square_test(reference, current), "chi2"
    return FeatureDrift(
        feature=str(reference.name),
        kind=kind,
        model_input=model_input,
        psi=round(psi, 5),
        test=test,
        statistic=round(result.statistic, 5),
        p_value=result.p_value,
        reference_mean=float(reference.mean()),
        current_mean=float(current.mean()),
        severity=classify(psi, result.statistic, result.p_value, thresholds),
    )


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    thresholds: DriftThresholds | None = None,
    numerical: tuple[str, ...] = NUMERICAL_MONITORED,
    categorical: tuple[str, ...] = CATEGORICAL_MONITORED,
) -> DriftReport:
    """Roda PSI + KS/qui-quadrado em todas as features monitoradas presentes nos dois datasets."""
    thresholds = thresholds or DriftThresholds()
    report = DriftReport(thresholds=thresholds)
    for kind, columns in (("numerical", numerical), ("categorical", categorical)):
        for column in columns:
            if column in reference.columns and column in current.columns:
                report.features.append(
                    detect_feature_drift(
                        reference[column], current[column], kind, thresholds, model_input=column in MODEL_FEATURES
                    )
                )
    return report
