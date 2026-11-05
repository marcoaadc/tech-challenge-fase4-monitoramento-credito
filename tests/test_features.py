from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditwatch.data.schema import BILL_AMT, ENGINEERED, MODEL_FEATURES, PAY_AMT, PAY_STATUS
from creditwatch.features.engineering import (
    PAYMENT_RATIO_BOUNDS,
    UTILIZATION_BOUNDS,
    add_engineered_features,
    model_matrix,
)


def _customer(**overrides) -> pd.DataFrame:
    row = {
        "limit_bal": 100_000,
        **dict.fromkeys(PAY_STATUS, 0),
        **dict.fromkeys(BILL_AMT, 20000),
        **dict.fromkeys(PAY_AMT, 5000),
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_engineered_values_for_a_known_customer():
    features = add_engineered_features(
        _customer(bill_amt_1=50_000, bill_amt_6=10_000, pay_status_1=2, pay_status_2=1, pay_status_3=-1)
    ).iloc[0]
    assert features["utilization_1"] == pytest.approx(0.5)
    assert features["utilization_avg"] == pytest.approx((50_000 + 4 * 20_000 + 10_000) / 6 / 100_000)
    assert features["payment_ratio_1"] == pytest.approx(5_000 / 20_000)
    assert features["max_delay"] == 2
    assert features["months_delayed"] == 2
    assert features["bill_trend"] == pytest.approx(0.4)


def test_customer_without_balance_counts_as_fully_paid():
    features = add_engineered_features(_customer(**dict.fromkeys(BILL_AMT, 0))).iloc[0]
    assert features["payment_ratio_1"] == 1.0
    assert features["payment_ratio_avg"] == 1.0
    assert features["utilization_1"] == 0.0


def test_customer_who_never_delayed_has_zero_delay_features():
    features = add_engineered_features(_customer(**dict.fromkeys(PAY_STATUS, -2))).iloc[0]
    assert features["max_delay"] == 0
    assert features["months_delayed"] == 0


def test_ratios_are_clipped_to_their_bounds():
    extreme = add_engineered_features(
        pd.concat([_customer(bill_amt_1=5_000_000, pay_amt_1=900_000), _customer(bill_amt_1=-900_000)])
    )
    assert extreme["utilization_1"].max() == UTILIZATION_BOUNDS[1]
    assert extreme["utilization_1"].min() == UTILIZATION_BOUNDS[0]
    assert extreme["payment_ratio_1"].max() == PAYMENT_RATIO_BOUNDS[1]


def test_payment_ratio_is_invariant_to_uniform_inflation():
    base = add_engineered_features(_customer()).iloc[0]
    inflated_customer = _customer(**dict.fromkeys(BILL_AMT, 40000), **dict.fromkeys(PAY_AMT, 10000))
    inflated = add_engineered_features(inflated_customer).iloc[0]
    assert inflated["payment_ratio_1"] == pytest.approx(base["payment_ratio_1"])
    assert inflated["utilization_1"] == pytest.approx(2 * base["utilization_1"])


def test_real_data_produces_finite_features(clean_frame):
    enriched = add_engineered_features(clean_frame)
    assert set(ENGINEERED) <= set(enriched.columns)
    assert np.isfinite(enriched[list(ENGINEERED)].to_numpy()).all()
    assert len(enriched) == len(clean_frame)
    assert "utilization_1" not in clean_frame.columns  # a entrada nao e alterada


def test_model_matrix_has_fixed_column_order(clean_frame):
    sample = clean_frame.head(200)
    from_raw = model_matrix(sample)
    from_enriched = model_matrix(add_engineered_features(sample))
    assert list(from_raw.columns) == list(MODEL_FEATURES)
    pd.testing.assert_frame_equal(from_raw, from_enriched)
    assert (from_raw.dtypes == "float64").all()
