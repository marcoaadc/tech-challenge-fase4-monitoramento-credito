"""Portao de qualidade: valida um lote contra o contrato e bloqueia a ingestao se ele reprovar.

O lote reprovado nao segue para o modelo: e movido para a quarentena junto com um relatorio que
diz *qual regra* falhou, *em quantas linhas* e *em quais colunas*. O relatorio guarda exemplos de
valores apenas das colunas previstas no contrato; de colunas inesperadas (possivel PII) registra
somente o nome.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors, SchemaWarning

from creditwatch.contracts.rules import RULES_BY_ID, rule_id_for_check
from creditwatch.data.schema import ID_COLUMN

logger = logging.getLogger("creditwatch.contracts.gate")

MAX_EXAMPLES = 5
SCHEMA_RULE = "DQ-09"
TYPE_RULE = "DQ-10"


class DataContractError(ValueError):
    """O lote violou o contrato de dados e foi bloqueado."""

    def __init__(self, result: ContractResult):
        self.result = result
        rules = ", ".join(f"{v.rule_id} ({v.n_failures})" for v in result.violations)
        super().__init__(f"lote '{result.batch}' reprovado no contrato de dados: {rules}")


@dataclass(frozen=True)
class Violation:
    rule_id: str
    title: str
    columns: list[str]
    n_failures: int
    examples: list[str]

    def as_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "title": self.title,
            "description": RULES_BY_ID[self.rule_id].description if self.rule_id in RULES_BY_ID else "",
            "columns": self.columns,
            "n_failures": self.n_failures,
            "examples": self.examples,
        }


@dataclass
class ContractResult:
    batch: str
    contract: str
    n_rows: int
    n_columns: int
    violations: list[Violation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rows_with_failures: int = 0
    validated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).replace(microsecond=0).isoformat())

    @property
    def passed(self) -> bool:
        return not self.violations

    @property
    def violated_rules(self) -> list[str]:
        return [v.rule_id for v in self.violations]

    def as_dict(self) -> dict:
        return {
            "batch": self.batch,
            "contract": self.contract,
            "status": "aprovado" if self.passed else "bloqueado",
            "validated_at": self.validated_at,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "rows_with_failures": self.rows_with_failures,
            "violations": [v.as_dict() for v in self.violations],
            "warnings": self.warnings,
        }


def _is_execution_error(row: pd.Series) -> bool:
    """Check que nem chegou a rodar (ex.: comparar texto com numero): efeito colateral de outra falha."""
    return row["schema_context"] == "Column" and pd.isna(row["index"]) and "Error(" in str(row["failure_case"])


def _format_example(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _summarize(failure_cases: pd.DataFrame, contract_columns: set[str]) -> tuple[list[Violation], int]:
    """Agrupa os casos de falha do Pandera por regra do catalogo."""
    cases = failure_cases.copy()
    cases = cases[~cases.apply(_is_execution_error, axis=1)]
    cases["rule_id"] = cases["check"].map(rule_id_for_check)
    # Um nulo tambem dispara falha de tipo na mesma celula: conta apenas como completude (DQ-01).
    cases = cases[~((cases["rule_id"] == TYPE_RULE) & cases["failure_case"].isna())]

    violations: list[Violation] = []
    for rule_id, group in cases.groupby("rule_id", sort=True):
        if rule_id == SCHEMA_RULE:
            # Para falhas de esquema, o "valor" e o nome da coluna ausente ou inesperada.
            columns = sorted({str(value) for value in group["failure_case"]})
            examples: list[str] = []
            n_failures = len(columns)
        else:
            column_level = group[group["schema_context"] == "Column"]
            columns = sorted({str(c) for c in column_level["column"].dropna()})
            deduplicated = group.drop_duplicates(subset=["column", "index"])
            n_failures = len(deduplicated)
            # Exemplos so de colunas do contrato e nunca do identificador do cliente.
            shareable = deduplicated["column"].isin(contract_columns - {ID_COLUMN})
            values = deduplicated.loc[shareable & (deduplicated["schema_context"] == "Column"), "failure_case"]
            examples = [_format_example(v) for v in values.dropna().unique()[:MAX_EXAMPLES]]
        title = RULES_BY_ID[rule_id].title if rule_id in RULES_BY_ID else str(group["check"].iloc[0])
        violations.append(Violation(rule_id, title, columns, int(n_failures), examples))

    row_level = cases[cases["index"].notna()]
    return violations, int(row_level["index"].nunique())


def validate_batch(frame: pd.DataFrame, contract: pa.DataFrameSchema, batch: str) -> ContractResult:
    """Valida ``frame`` e devolve o resultado, sem levantar excecao e sem alterar o lote."""
    result = ContractResult(batch=batch, contract=str(contract.name), n_rows=len(frame), n_columns=frame.shape[1])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            contract.validate(frame, lazy=True)
        except SchemaErrors as exc:
            result.violations, result.rows_with_failures = _summarize(exc.failure_cases, set(contract.columns))
    for warning in caught:
        if issubclass(warning.category, SchemaWarning):
            message = str(warning.message)
            result.warnings.append(message[message.find("DQ-W") :].rstrip(">") if "DQ-W" in message else message)
    return result


def write_validation_report(result: ContractResult, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result.as_dict(), indent=2, ensure_ascii=False) + "\n"
    path.write_text(payload, encoding="utf-8", newline="\n")
    return path


def quarantine_batch(frame: pd.DataFrame, result: ContractResult, quarantine_dir: Path) -> Path:
    """Grava o lote reprovado e o seu relatorio na quarentena (acesso restrito, retencao curta)."""
    target = Path(quarantine_dir) / result.batch
    target.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target / "batch.csv", index=False, lineterminator="\n")
    write_validation_report(result, target / "validation_report.json")
    return target


def enforce_contract(
    frame: pd.DataFrame, contract: pa.DataFrameSchema, batch: str, quarantine_dir: Path | None = None
) -> tuple[pd.DataFrame, ContractResult]:
    """Valida o lote; se reprovar, coloca em quarentena e levanta ``DataContractError``.

    Em caso de sucesso devolve o lote com os tipos do contrato aplicados.
    """
    result = validate_batch(frame, contract, batch)
    if not result.passed:
        if quarantine_dir is not None:
            location = quarantine_batch(frame, result, quarantine_dir)
            logger.error("lote %s BLOQUEADO e movido para %s", batch, location)
        raise DataContractError(result)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SchemaWarning)
        validated = contract.validate(frame)
    return validated, result
