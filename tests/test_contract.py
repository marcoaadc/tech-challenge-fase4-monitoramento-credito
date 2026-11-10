from __future__ import annotations

import json

import pandas as pd
import pytest

from creditwatch.contracts.credit_contract import build_contract, build_label_contract
from creditwatch.contracts.gate import (
    DataContractError,
    enforce_contract,
    quarantine_batch,
    validate_batch,
    write_validation_report,
)
from creditwatch.contracts.rules import BLOCKING, RULES, RULES_BY_ID, WARNING, rule_id_for_check
from creditwatch.data.schema import ID_COLUMN, PAY_STATUS, PAY_STATUS_MIN, TARGET
from creditwatch.simulation.corruption import (
    DEMO_DEFECTS,
    INJECTORS,
    corrupt_batch,
    cpf_check_digits,
    expected_rules,
    fake_cpf,
)

PRODUCTION = build_contract(require_target=False)
REFERENCE = build_contract(require_target=True)


def test_rule_catalog_has_unique_ids_and_at_least_three_blocking_rules():
    ids = [rule.rule_id for rule in RULES]
    assert len(ids) == len(set(ids))
    assert sum(rule.severity == BLOCKING for rule in RULES) >= 3
    assert RULES_BY_ID["DQ-W01"].severity == WARNING
    assert RULES_BY_ID["DQ-02"].as_dict()["title"] == "Idade válida"


@pytest.mark.parametrize(
    ("check", "rule"),
    [
        ("DQ-02: Idade válida", "DQ-02"),
        ("not_nullable", "DQ-01"),
        ("field_uniqueness", "DQ-04"),
        ("column_in_schema", "DQ-09"),
        ("column_in_dataframe", "DQ-09"),
        ("coerce_dtype('int64')", "DQ-10"),
        ("dtype('int64')", "DQ-10"),
        ("algo_inesperado", "DQ-??"),
    ],
)
def test_rule_id_for_check(check, rule):
    assert rule_id_for_check(check) == rule


def test_clean_reference_passes_the_contract(clean_frame):
    result = validate_batch(clean_frame, REFERENCE, "reference")
    assert result.passed
    assert result.violations == []
    assert result.as_dict()["status"] == "aprovado"


def test_raw_dataset_is_rejected_for_undocumented_categories(raw_frame):
    result = validate_batch(raw_frame, REFERENCE, "raw")
    assert not result.passed
    assert result.violated_rules == ["DQ-05"]
    violation = result.violations[0]
    assert violation.columns == ["education", "marriage"]
    assert violation.n_failures == result.rows_with_failures == 399


def test_valid_production_batch_passes(production_features):
    assert validate_batch(production_features, PRODUCTION, "2025-01").passed


@pytest.mark.parametrize("defect", sorted(INJECTORS))
def test_each_injected_defect_breaks_exactly_its_rule(production_features, defect):
    expected_rule = INJECTORS[defect][0]
    result = validate_batch(corrupt_batch(production_features, (defect,)), PRODUCTION, defect)
    assert result.violated_rules == [expected_rule]
    assert result.violations[0].n_failures > 0


def test_demo_batch_breaks_every_required_rule(production_features):
    result = validate_batch(corrupt_batch(production_features), PRODUCTION, "demo")
    assert result.violated_rules == expected_rules(DEMO_DEFECTS)
    # Regras exigidas no enunciado: nulos, faixa de idade, limites e duplicatas.
    assert {"DQ-01", "DQ-02", "DQ-03", "DQ-04"} <= set(result.violated_rules)
    assert result.rows_with_failures > 0


def test_corrupt_batch_is_deterministic_and_does_not_mutate_the_input(production_features):
    before = production_features.copy()
    first = corrupt_batch(production_features, seed=1)
    second = corrupt_batch(production_features, seed=1)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(production_features, before)


def test_corrupt_batch_rejects_unknown_defects(production_features):
    with pytest.raises(ValueError, match="defeitos desconhecidos"):
        corrupt_batch(production_features, ("nao_existe",))


def test_strict_schema_blocks_pii_without_exposing_its_values(production_features):
    leaked = corrupt_batch(production_features, ("leaked_pii_column",))
    result = validate_batch(leaked, PRODUCTION, "pii")
    violation = result.violations[0]
    assert violation.rule_id == "DQ-09"
    assert violation.columns == ["cpf"]
    assert violation.examples == []
    serialized = json.dumps(result.as_dict())
    assert not any(cpf in serialized for cpf in leaked["cpf"].head(50))


def test_duplicate_report_does_not_expose_customer_ids(production_features):
    result = validate_batch(corrupt_batch(production_features, ("duplicated_customers",)), PRODUCTION, "dup")
    assert result.violations[0].examples == []


def test_examples_show_invalid_values_of_contract_columns(production_features):
    result = validate_batch(corrupt_batch(production_features, ("invalid_ages",)), PRODUCTION, "ages")
    examples = [int(value) for value in result.violations[0].examples]
    assert examples
    assert all(value < 18 or value > 100 for value in examples)


def test_production_contract_rejects_the_label_column(stable_batch):
    result = validate_batch(stable_batch, PRODUCTION, "with-label")
    assert result.violated_rules == ["DQ-09"]
    assert result.violations[0].columns == [TARGET]


def test_reference_contract_requires_a_binary_label(clean_frame):
    sample = clean_frame.head(1000)
    assert validate_batch(sample.drop(columns=[TARGET]), REFERENCE, "no-label").violated_rules == ["DQ-09"]
    assert validate_batch(sample.assign(**{TARGET: 2}), REFERENCE, "bad-label").violated_rules == ["DQ-12"]


def test_batch_with_two_reference_months_is_rejected(production_features):
    mixed = production_features.copy()
    mixed.loc[mixed.index[:10], "reference_month"] = "2025-02"
    assert validate_batch(mixed, PRODUCTION, "mixed").violated_rules == ["DQ-13"]


def test_malformed_reference_month_is_rejected(production_features):
    result = validate_batch(production_features.assign(reference_month="jan/2025"), PRODUCTION, "month")
    assert result.violated_rules == ["DQ-13"]


def test_soft_rule_only_warns(production_features):
    inactive = production_features.copy()
    inactive[PAY_STATUS[0]] = PAY_STATUS_MIN
    result = validate_batch(inactive, PRODUCTION, "inactive")
    assert result.passed
    assert len(result.warnings) == 1
    assert result.warnings[0].startswith("DQ-W01")


def test_enforce_contract_returns_typed_batch(production_features):
    as_float = production_features.astype({"age": "float64"})
    validated, result = enforce_contract(as_float, PRODUCTION, "typed")
    assert result.passed
    assert str(validated["age"].dtype) == "int64"


def test_enforce_contract_blocks_and_quarantines(production_features, tmp_path):
    bad = corrupt_batch(production_features)
    with pytest.raises(DataContractError) as excinfo:
        enforce_contract(bad, PRODUCTION, "bad-batch", quarantine_dir=tmp_path)
    assert "DQ-01" in str(excinfo.value)
    assert not excinfo.value.result.passed
    quarantined = tmp_path / "bad-batch"
    assert len(pd.read_csv(quarantined / "batch.csv")) == len(bad)
    report = json.loads((quarantined / "validation_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "bloqueado"
    assert [violation["rule_id"] for violation in report["violations"]] == expected_rules()


def test_enforce_contract_without_quarantine_dir_still_blocks(production_features):
    with pytest.raises(DataContractError):
        enforce_contract(corrupt_batch(production_features, ("null_limits",)), PRODUCTION, "no-quarantine")


def test_validation_report_and_quarantine_helpers(production_features, tmp_path):
    result = validate_batch(production_features, PRODUCTION, "ok")
    path = write_validation_report(result, tmp_path / "nested" / "report.json")
    assert json.loads(path.read_text(encoding="utf-8"))["n_rows"] == len(production_features)
    assert (quarantine_batch(production_features, result, tmp_path) / "batch.csv").exists()


def test_label_contract(stable_batch):
    labels = stable_batch[[ID_COLUMN, TARGET]]
    contract = build_label_contract()
    assert validate_batch(labels, contract, "labels").passed
    assert validate_batch(labels.assign(**{TARGET: 3}), contract, "labels").violated_rules == ["DQ-12"]
    duplicated = pd.concat([labels, labels.head(5)])
    assert validate_batch(duplicated, contract, "labels").violated_rules == ["DQ-04"]


def test_demo_cpfs_are_never_valid():
    import numpy as np

    assert cpf_check_digits("529982247") == "25"  # CPF de exemplo amplamente usado em documentacao
    rng = np.random.default_rng(0)
    for _ in range(500):
        cpf = fake_cpf(rng)
        assert len(cpf) == 11
        assert cpf[9:] != cpf_check_digits(cpf[:9])


def test_fraction_in_integer_column_is_rejected_instead_of_truncated(production_features):
    """Sem a checagem, a coercao de tipos transformaria 0.6 em 0 e o lote passaria."""
    fractional = production_features.astype({"pay_status_1": "float64", "limit_bal": "float64"})
    fractional.loc[fractional.index[:3], "pay_status_1"] = 0.6
    fractional.loc[fractional.index[5], "limit_bal"] = 50_000.7
    result = validate_batch(fractional, PRODUCTION, "fraction")
    assert result.violated_rules == ["DQ-10"]
    violation = result.violations[0]
    assert violation.columns == ["limit_bal", "pay_status_1"]
    assert violation.n_failures == 4
    assert result.rows_with_failures == 4
    with pytest.raises(DataContractError):
        enforce_contract(fractional, PRODUCTION, "fraction")


def test_float_column_with_whole_numbers_still_passes(production_features):
    as_float = production_features.astype({"age": "float64", "bill_amt_1": "float64"})
    assert validate_batch(as_float, PRODUCTION, "floats").passed


def test_quarantine_does_not_keep_columns_outside_the_contract(production_features, tmp_path):
    """Um dado pessoal enviado por engano nao pode ganhar uma copia na quarentena."""
    leaked = corrupt_batch(production_features, ("leaked_pii_column", "invalid_ages"))
    with pytest.raises(DataContractError):
        enforce_contract(leaked, PRODUCTION, "pii-batch", quarantine_dir=tmp_path)
    quarantined = pd.read_csv(tmp_path / "pii-batch" / "batch.csv")
    assert "cpf" not in quarantined.columns
    assert len(quarantined) == len(leaked)
    assert set(quarantined.columns) == set(production_features.columns)
    text = (tmp_path / "pii-batch" / "batch.csv").read_text(encoding="utf-8")
    assert not any(cpf in text for cpf in leaked["cpf"].head(50))
    report = json.loads((tmp_path / "pii-batch" / "validation_report.json").read_text(encoding="utf-8"))
    assert "cpf" in next(v for v in report["violations"] if v["rule_id"] == "DQ-09")["columns"]
