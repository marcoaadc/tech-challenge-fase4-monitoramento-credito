from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from creditwatch.data import source
from creditwatch.data.cleaning import clean_raw_dataset
from creditwatch.data.schema import (
    CONTRACT_COLUMNS,
    EDUCATION_DOMAIN,
    ID_COLUMN,
    MARRIAGE_DOMAIN,
    MODEL_FEATURES,
    MONITORED,
    PROTECTED,
    RAW_RENAME,
    TARGET,
)
from creditwatch.data.split import SPLIT_COLUMN, assign_model_splits, split_reference_production


def test_schema_keeps_protected_attributes_out_of_the_model():
    assert not set(PROTECTED) & set(MODEL_FEATURES)
    assert set(PROTECTED) <= set(CONTRACT_COLUMNS)
    assert set(MODEL_FEATURES) <= set(MONITORED)
    assert len(MODEL_FEATURES) == len(set(MODEL_FEATURES)) == 26


def test_raw_dataset_is_loaded_with_normalized_names(raw_frame):
    assert list(raw_frame.columns) == list(RAW_RENAME.values())
    assert len(raw_frame) == source.EXPECTED_ROWS
    assert raw_frame[ID_COLUMN].is_unique
    assert "pay_status_1" in raw_frame.columns  # PAY_0 do arquivo original


def test_load_raw_dataset_requires_the_file_when_download_is_disabled(tmp_path):
    with pytest.raises(FileNotFoundError):
        source.load_raw_dataset(tmp_path / "missing.csv", download_if_missing=False)


def test_load_raw_dataset_rejects_wrong_layout(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"ID": [1], "AGE": [30]}).to_csv(path, index=False)
    with pytest.raises(source.DatasetIntegrityError, match="colunas ausentes"):
        source.load_raw_dataset(path)


def test_load_raw_dataset_rejects_truncated_file(tmp_path, raw_frame):
    path = tmp_path / "short.csv"
    inverse = {new: old for old, new in RAW_RENAME.items()}
    raw_frame.head(10).rename(columns=inverse).to_csv(path, index=False)
    with pytest.raises(source.DatasetIntegrityError, match="esperadas 30000 linhas"):
        source.load_raw_dataset(path)


def test_load_raw_dataset_rejects_an_edited_snapshot(tmp_path, raw_frame):
    """Mesmo layout e mesmas 30.000 linhas, mas um valor alterado: o hash acusa."""
    inverse = {new: old for old, new in RAW_RENAME.items()}
    edited = raw_frame.rename(columns=inverse)
    edited.loc[0, "LIMIT_BAL"] += 1
    path = tmp_path / "edited.csv"
    edited.to_csv(path, index=False, lineterminator="\n")
    with pytest.raises(source.DatasetIntegrityError, match="sha256 do CSV bruto"):
        source.load_raw_dataset(path)
    assert len(source.load_raw_dataset(path, expected_sha256=None)) == source.EXPECTED_ROWS


def test_versioned_snapshot_matches_the_pinned_hash():
    snapshot = Path(__file__).resolve().parents[1] / "data" / "raw" / "default_of_credit_card_clients.csv"
    assert source.sha256_of(snapshot) == source.RAW_CSV_SHA256


def _fake_zip(frame: pd.DataFrame, monkeypatch) -> bytes:
    """Zip com um .xls ficticio; a leitura do Excel e substituida para nao depender do formato."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("default of credit card clients.xls", b"conteudo")
    monkeypatch.setattr(source.pd, "read_excel", lambda *_args, **_kwargs: frame)
    return buffer.getvalue()


def test_fetch_raw_dataset_verifies_the_hash_before_writing(tmp_path, monkeypatch):
    payload = _fake_zip(pd.DataFrame({"ID": [1, 2]}), monkeypatch)
    monkeypatch.setattr(source, "_download", lambda *_args, **_kwargs: payload)
    target = tmp_path / "raw" / "dataset.csv"

    with pytest.raises(source.DatasetIntegrityError, match="sha256"):
        source.fetch_raw_dataset(target, expected_sha256="0" * 64)
    assert not target.exists()

    source.fetch_raw_dataset(target, expected_sha256=hashlib.sha256(payload).hexdigest())
    assert pd.read_csv(target)["ID"].tolist() == [1, 2]


def test_sha256_of_matches_hashlib(tmp_path):
    path = tmp_path / "file.bin"
    path.write_bytes(b"creditwatch" * 1000)
    assert source.sha256_of(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_cleaning_recodes_undocumented_categories_and_removes_duplicates(raw_frame):
    cleaned, report = clean_raw_dataset(raw_frame)
    assert set(cleaned["education"].unique()) <= set(EDUCATION_DOMAIN)
    assert set(cleaned["marriage"].unique()) <= set(MARRIAGE_DOMAIN)
    assert report.education_recoded == 345
    assert report.marriage_recoded == 54
    assert report.duplicates_removed == 35
    assert report.rows_out == report.rows_in - report.duplicates_removed == len(cleaned)
    assert not cleaned.drop(columns=[ID_COLUMN]).duplicated().any()
    assert report.as_dict()["rows_in"] == 30_000


def test_cleaning_does_not_mutate_its_input(raw_frame):
    before = raw_frame["education"].value_counts().to_dict()
    clean_raw_dataset(raw_frame)
    assert raw_frame["education"].value_counts().to_dict() == before


def test_reference_and_production_are_disjoint_and_stratified(clean_frame):
    reference, pool = split_reference_production(clean_frame, 0.5, seed=42)
    assert not set(reference[ID_COLUMN]) & set(pool[ID_COLUMN])
    assert len(reference) + len(pool) == len(clean_frame)
    assert reference[TARGET].mean() == pytest.approx(pool[TARGET].mean(), abs=0.001)


def test_split_is_deterministic_by_seed(clean_frame):
    first, _ = split_reference_production(clean_frame, 0.5, seed=42)
    second, _ = split_reference_production(clean_frame, 0.5, seed=42)
    other, _ = split_reference_production(clean_frame, 0.5, seed=7)
    assert first[ID_COLUMN].tolist() == second[ID_COLUMN].tolist()
    assert first[ID_COLUMN].tolist() != other[ID_COLUMN].tolist()


def test_model_splits_cover_the_reference_with_the_requested_proportions(reference):
    shares = reference[SPLIT_COLUMN].value_counts(normalize=True)
    assert set(shares.index) == {"train", "val", "test"}
    assert shares["train"] == pytest.approx(0.6, abs=0.01)
    assert shares["val"] == pytest.approx(0.2, abs=0.01)
    rates = reference.groupby(SPLIT_COLUMN)[TARGET].mean()
    assert rates.max() - rates.min() < 0.005


def test_assign_model_splits_keeps_every_row_once(clean_frame):
    reference, _ = split_reference_production(clean_frame)
    marked = assign_model_splits(reference)
    assert sorted(marked[ID_COLUMN]) == sorted(reference[ID_COLUMN])
