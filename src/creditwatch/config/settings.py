"""Configuracao centralizada (12-factor: tudo via variaveis de ambiente).

Todas as chaves possuem defaults para execucao local; em CI os valores sao sobrescritos por
variaveis de ambiente (prefixo ``CREDITWATCH_``) ou por um arquivo ``.env``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CREDITWATCH_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Diretorios --------------------------------------------------------
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")
    logs_dir: Path = Path("logs")

    # --- Reprodutibilidade -------------------------------------------------
    seed: int = 42
    log_level: str = "INFO"

    # --- Particao referencia x producao -----------------------------------
    reference_fraction: float = Field(
        default=0.5, gt=0.1, lt=0.9, description="Fracao do dataset limpo reservada como Dataset de Referencia."
    )
    batch_size: int = Field(default=2000, ge=500, description="Propostas por lote mensal de producao.")
    label_lag_months: int = Field(
        default=2, ge=0, description="Meses ate o rotulo (inadimplencia) de um lote ficar disponivel."
    )

    # --- Limiares de drift (docs/observabilidade.md) --------------------------
    psi_warning: float = Field(default=0.10, gt=0)
    psi_critical: float = Field(default=0.25, gt=0)
    ks_effect_size: float = Field(
        default=0.10, gt=0, description="Estatistica D minima do KS para considerar o drift relevante."
    )
    p_value_alpha: float = Field(default=0.05, gt=0, lt=1)
    drift_share_alert: float = Field(
        default=0.30, gt=0, le=1, description="Fracao de features com drift que caracteriza drift do dataset."
    )

    # --- Limiares de degradacao do modelo ----------------------------------
    auc_drop_warning: float = Field(default=0.03, gt=0)
    auc_drop_critical: float = Field(default=0.05, gt=0)
    calibration_gap_alert: float = Field(
        default=0.05, gt=0, description="Diferenca absoluta entre inadimplencia observada e PD media prevista."
    )

    # --- Observabilidade ---------------------------------------------------
    mlflow_tracking_uri: str | None = Field(
        default=None, description="Se definido, treino e monitoramento sao rastreados no MLflow."
    )
    mlflow_experiment_training: str = "creditwatch-treino"
    mlflow_experiment_monitoring: str = "creditwatch-monitoramento"
    mlflow_registered_model: str = "credit-scoring-baseline"
    pushgateway_url: str | None = Field(
        default=None, description="Se definido, as metricas tambem sao enviadas a um Prometheus Pushgateway."
    )

    # --- Caminhos derivados ------------------------------------------------
    @property
    def raw_dataset_path(self) -> Path:
        return self.data_dir / "raw" / "default_of_credit_card_clients.csv"

    @property
    def reference_path(self) -> Path:
        return self.data_dir / "reference" / "reference.parquet"

    @property
    def production_pool_path(self) -> Path:
        return self.data_dir / "interim" / "production_pool.parquet"

    @property
    def production_dir(self) -> Path:
        return self.data_dir / "production"

    @property
    def quarantine_dir(self) -> Path:
        return self.data_dir / "quarantine"

    @property
    def observability_dir(self) -> Path:
        return self.reports_dir / "observability"

    @property
    def events_path(self) -> Path:
        return self.observability_dir / "pipeline_events.jsonl"

    @property
    def alerts_path(self) -> Path:
        return self.observability_dir / "alerts.jsonl"


@lru_cache
def get_settings() -> Settings:
    """Instancia unica (cacheada) das configuracoes."""
    return Settings()
