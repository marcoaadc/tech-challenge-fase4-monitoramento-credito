"""Fixtures compartilhadas.

Os testes usam o snapshot real do dataset (versionado em ``data/raw``). A fixture ``workspace``
executa o pipeline completo uma unica vez por sessao em um diretorio temporario, para que os
testes de integracao verifiquem artefatos verdadeiros sem tocar nos arquivos do repositorio.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from creditwatch.config.settings import Settings
from creditwatch.data.cleaning import clean_raw_dataset
from creditwatch.data.schema import TARGET
from creditwatch.data.source import load_raw_dataset
from creditwatch.data.split import assign_model_splits, split_reference_production
from creditwatch.observability.events import StageTracker
from creditwatch.observability.logs import new_run_id
from creditwatch.pipelines.analyze import run_analyze
from creditwatch.pipelines.dashboard import run_dashboard
from creditwatch.pipelines.ingest import run_ingest
from creditwatch.pipelines.monitor import run_monitor
from creditwatch.pipelines.simulate import run_simulate
from creditwatch.pipelines.train import run_train
from creditwatch.simulation.scenario import default_scenario, generate_batch

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DATASET = REPO_ROOT / "data" / "raw" / "default_of_credit_card_clients.csv"


@pytest.fixture(autouse=True)
def _restore_logging():
    """Evita que handlers configurados por um teste (ex.: arquivo de log) vazem para os demais."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    for handler in root.handlers:
        if handler not in handlers:
            handler.close()
    root.handlers[:] = handlers
    root.setLevel(level)


def make_settings(root: Path, **overrides) -> Settings:
    """Settings isoladas em ``root`` (sem ler ``.env`` nem variaveis do ambiente do desenvolvedor)."""
    values = {
        "data_dir": root / "data",
        "models_dir": root / "models",
        "reports_dir": root / "reports",
        "logs_dir": root / "logs",
        "batch_size": 1000,
        "mlflow_tracking_uri": None,
        "pushgateway_url": None,
        **overrides,
    }
    return Settings(_env_file=None, **values)


@pytest.fixture
def isolated_settings(tmp_path: Path) -> Settings:
    """Settings apontando para um diretorio temporario vazio."""
    return make_settings(tmp_path)


@pytest.fixture(scope="session")
def raw_frame() -> pd.DataFrame:
    return load_raw_dataset(RAW_DATASET, download_if_missing=False)


@pytest.fixture(scope="session")
def clean_frame(raw_frame: pd.DataFrame) -> pd.DataFrame:
    return clean_raw_dataset(raw_frame)[0]


@pytest.fixture(scope="session")
def reference(clean_frame: pd.DataFrame) -> pd.DataFrame:
    return assign_model_splits(split_reference_production(clean_frame)[0])


@pytest.fixture(scope="session")
def pool(clean_frame: pd.DataFrame) -> pd.DataFrame:
    return split_reference_production(clean_frame)[1]


@pytest.fixture(scope="session")
def stable_batch(pool: pd.DataFrame) -> pd.DataFrame:
    """Lote sem nenhum mecanismo de drift, com rotulo."""
    return generate_batch(pool, default_scenario()[0], 2000)


@pytest.fixture(scope="session")
def production_features(stable_batch: pd.DataFrame) -> pd.DataFrame:
    """Lote valido como chega para escoragem (sem rotulo)."""
    return stable_batch.drop(columns=[TARGET])


@dataclass
class Workspace:
    settings: Settings
    overview: dict
    analysis: dict
    dashboard: Path


@pytest.fixture(scope="session")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Workspace:
    """Pipeline completo executado uma vez em diretorio temporario (sem Evidently, sem MLflow)."""
    root = tmp_path_factory.mktemp("workspace")
    settings = make_settings(root)
    settings.raw_dataset_path.parent.mkdir(parents=True)
    shutil.copy2(RAW_DATASET, settings.raw_dataset_path)
    new_run_id()
    run_ingest(settings, StageTracker("ingest", settings.events_path))
    run_train(settings, StageTracker("train", settings.events_path))
    run_simulate(settings, StageTracker("simulate", settings.events_path))
    overview = run_monitor(settings, StageTracker("monitor", settings.events_path), evidently="none")
    analysis = run_analyze(settings, StageTracker("analyze", settings.events_path), attribution_seeds=2)
    dashboard = run_dashboard(settings, StageTracker("dashboard", settings.events_path))
    return Workspace(settings=settings, overview=overview, analysis=analysis, dashboard=dashboard)
