from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from creditwatch.config.settings import Settings, get_settings
from creditwatch.pipelines._cli import read_json, read_table, write_json, write_table


def test_defaults_point_to_the_repository_layout():
    settings = Settings(_env_file=None)
    assert settings.raw_dataset_path == Path("data/raw/default_of_credit_card_clients.csv")
    assert settings.reference_path == Path("data/reference/reference.parquet")
    assert settings.production_dir == Path("data/production")
    assert settings.quarantine_dir == Path("data/quarantine")
    assert settings.events_path == Path("reports/observability/pipeline_events.jsonl")
    assert settings.alerts_path == Path("reports/observability/alerts.jsonl")
    assert settings.production_pool_path.parent != settings.reference_path.parent
    assert (settings.psi_warning, settings.psi_critical) == (0.10, 0.25)
    assert settings.mlflow_tracking_uri is None


def test_environment_variables_override_defaults(monkeypatch):
    monkeypatch.setenv("CREDITWATCH_PSI_CRITICAL", "0.3")
    monkeypatch.setenv("CREDITWATCH_DATA_DIR", "/tmp/dados")
    monkeypatch.setenv("CREDITWATCH_MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    settings = Settings(_env_file=None)
    assert settings.psi_critical == 0.3
    assert settings.data_dir == Path("/tmp/dados")
    assert settings.mlflow_tracking_uri == "sqlite:///mlflow.db"


@pytest.mark.parametrize(
    "override",
    [
        {"batch_size": 10},
        {"reference_fraction": 0.95},
        {"psi_warning": 0},
        {"p_value_alpha": 1.5},
        {"label_lag_months": -1},
    ],
)
def test_invalid_values_are_rejected(override):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **override)


def test_get_settings_is_cached():
    assert get_settings() is get_settings()


def test_json_helpers_write_valid_json_for_nan_and_numpy(tmp_path):
    import numpy as np

    payload = {"auc": float("nan"), "n": np.int64(3), "rate": np.float64(0.25), "items": (1, float("inf")), 1: "x"}
    path = write_json(tmp_path / "nested" / "report.json", payload)
    assert "NaN" not in path.read_text(encoding="utf-8")
    assert b"\r\n" not in path.read_bytes()  # mesma saida em qualquer sistema operacional
    assert read_json(path) == {"auc": None, "n": 3, "rate": 0.25, "items": [1, None], "1": "x"}


def test_table_helpers_support_parquet_and_csv(tmp_path):
    import pandas as pd

    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    for name in ("table.parquet", "nested/table.csv"):
        pd.testing.assert_frame_equal(read_table(write_table(frame, tmp_path / name)), frame)
