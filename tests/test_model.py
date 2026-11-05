from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from creditwatch.data.schema import MODEL_FEATURES, PROTECTED, TARGET
from creditwatch.data.split import SPLIT_COLUMN
from creditwatch.model.metrics import best_ks_threshold, classification_metrics, ks_statistic, score_summary
from creditwatch.model.registry import METADATA_FILENAME, MODEL_FILENAME, CreditScorer, ModelMetadata, save_model
from creditwatch.model.training import (
    build_candidate,
    candidate_params,
    feature_importance,
    predict_default_probability,
    train_and_select,
)


@pytest.fixture(scope="module")
def trained(reference):
    """Regressao logistica treinada na referencia (rapida o bastante para testes de unidade)."""
    return train_and_select(reference, seed=42, candidates=("logreg",))


def test_ks_statistic_of_perfect_and_useless_scores():
    y = np.array([0] * 50 + [1] * 50)
    assert ks_statistic(y, y.astype(float)) == pytest.approx(1.0)
    assert ks_statistic(y, np.full(100, 0.5)) == pytest.approx(0.0)
    assert np.isnan(ks_statistic(np.zeros(10), np.linspace(0, 1, 10)))


def test_best_ks_threshold_separates_the_classes():
    y = np.array([0] * 50 + [1] * 50)
    score = np.concatenate([np.linspace(0.0, 0.4, 50), np.linspace(0.6, 1.0, 50)])
    assert 0.4 < best_ks_threshold(y, score) <= 0.6


def test_classification_metrics_of_a_perfect_model():
    y = np.array([0] * 80 + [1] * 20)
    metrics = classification_metrics(y, y.astype(float), threshold=0.5)
    assert metrics["roc_auc"] == metrics["ks"] == metrics["gini"] == pytest.approx(1.0)
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == pytest.approx(1.0)
    assert metrics["approval_rate"] == pytest.approx(0.8)
    assert metrics["bad_rate_approved"] == 0.0
    assert metrics["default_rate"] == pytest.approx(0.2)
    assert metrics["calibration_gap"] == pytest.approx(0.0)


def test_classification_metrics_with_a_single_class_do_not_fail():
    metrics = classification_metrics(np.zeros(10), np.linspace(0.1, 0.9, 10), threshold=0.0)
    assert np.isnan(metrics["roc_auc"])
    assert np.isnan(metrics["bad_rate_approved"])  # ninguem aprovado
    assert metrics["approval_rate"] == 0.0


def test_score_summary():
    summary = score_summary(np.array([0.1, 0.2, 0.6, 0.9]), threshold=0.5)
    assert summary == {"n": 4, "mean_score": pytest.approx(0.45), "approval_rate": 0.5}


def test_build_candidate_and_params():
    assert isinstance(build_candidate("logreg"), Pipeline)
    assert build_candidate("xgboost").get_params()["n_estimators"] == candidate_params("xgboost")["n_estimators"]
    assert "C" in candidate_params("logreg")
    with pytest.raises(ValueError, match="candidato desconhecido"):
        build_candidate("rede_neural")


def test_training_produces_a_useful_baseline(trained):
    assert trained.selected.name == "logreg"
    assert trained.test_metrics["roc_auc"] > 0.72
    assert trained.test_metrics["ks"] > 0.35
    assert 0 < trained.threshold < 1
    assert set(trained.split_sizes) == {"train", "val", "test"}
    assert abs(trained.test_metrics["calibration_gap"]) < 0.03


def test_training_requires_the_three_splits(reference):
    with pytest.raises(ValueError, match="particoes train, val e test"):
        train_and_select(reference[reference[SPLIT_COLUMN] != "test"], candidates=("logreg",))


def test_feature_importance_is_normalized_and_ranked(trained):
    importance = feature_importance(trained.model)
    assert set(importance) == set(MODEL_FEATURES)
    assert sum(importance.values()) == pytest.approx(1.0, abs=1e-3)
    values = list(importance.values())
    assert values == sorted(values, reverse=True)


def test_predictions_are_probabilities(trained, reference):
    score = predict_default_probability(trained.model, reference.head(500))
    assert score.shape == (500,)
    assert ((score >= 0) & (score <= 1)).all()


def test_save_and_load_model_roundtrip(trained, reference, tmp_path):
    reference_path = tmp_path / "reference.parquet"
    reference.to_parquet(reference_path, index=False)
    metadata = save_model(trained, reference_path, tmp_path / "models", seed=42)

    assert (tmp_path / "models" / MODEL_FILENAME).exists()
    stored = json.loads((tmp_path / "models" / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert stored["model_version"] == metadata.model_version
    assert stored["excluded_protected_attributes"] == list(PROTECTED)
    assert not set(PROTECTED) & set(stored["features"])
    assert len(stored["reference"]["sha256"]) == 64
    assert stored["metrics"]["test"]["roc_auc"] == pytest.approx(trained.test_metrics["roc_auc"], abs=1e-4)

    scorer = CreditScorer.load(tmp_path / "models")
    sample = reference.head(300)
    np.testing.assert_allclose(scorer.score(sample), predict_default_probability(trained.model, sample))
    assert scorer.threshold == metadata.threshold
    assert scorer.decide(np.array([0.0, 1.0])).tolist() == [True, False]
    assert ModelMetadata.load(tmp_path / "models").model_type == "logreg"


def test_loading_a_tampered_model_fails(trained, reference, tmp_path):
    reference_path = tmp_path / "reference.parquet"
    reference.to_parquet(reference_path, index=False)
    save_model(trained, reference_path, tmp_path / "models", seed=42)
    with (tmp_path / "models" / MODEL_FILENAME).open("ab") as handle:
        handle.write(b"adulterado")
    with pytest.raises(RuntimeError, match="adulterado"):
        CreditScorer.load(tmp_path / "models")
    assert CreditScorer.load(tmp_path / "models", verify_integrity=False) is not None


def test_model_ranks_delinquent_customers_as_riskier(trained, reference):
    sample = reference[reference[SPLIT_COLUMN] == "test"]
    score = predict_default_probability(trained.model, sample)
    delayed = sample["pay_status_1"] >= 2
    assert score[delayed].mean() > score[~delayed].mean() + 0.2
    assert sample.loc[delayed, TARGET].mean() > sample.loc[~delayed, TARGET].mean()
