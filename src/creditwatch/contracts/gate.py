"""Portao de qualidade: valida um lote contra o contrato e bloqueia a ingestao se ele reprovar.

O lote reprovado nao segue para o modelo: e copiado para a quarentena junto com um relatorio que
diz *qual regra* falhou, *em quantas linhas* e *em quais colunas*. O relatorio guarda exemplos de
valores apenas das colunas previstas no contrato; de colunas inesperadas (possivel PII) registra
somente o nome, e a copia em quarentena tambem descarta essas colunas.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors, SchemaWarning

from creditwatch.contracts.rules import RULES_BY_ID, rule_id_for_check
from creditwatch.data.schema import ID_COLUMN

logger = logging.getLogger("creditwatch.contracts.gate")

MAX_EXAMPLES = 5
SCHEMA_RULE = "DQ-09"
TYPE_RULE = "DQ-10"
FAILURE_COLUMNS = ["schema_context", "column", "check", "check_number", "failure_case", "index"]


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


def _fractional_cases(frame: pd.DataFrame, contract: pa.DataFrameSchema) -> pd.DataFrame:
    """Valores com fracao em colunas inteiras, no formato dos casos de falha do Pandera.

    O contrato coage os tipos (um CSV le inteiros como float quando ha nulos), e a coercao
    truncaria ``17.9`` para ``17`` sem acusar nada. Esta checagem roda antes e reporta a fracao
    como violacao de tipo (DQ-10).
    """
    rows: list[dict] = []
    for name, column in contract.columns.items():
        if name not in frame.columns or str(column.dtype) != "int64":
            continue
        values = pd.to_numeric(frame[name], errors="coerce")
        fractional = values[values.notna() & np.isfinite(values) & (values % 1 != 0)]
        rows.extend(
            {
                "schema_context": "Column",
                "column": name,
                "check": f"{TYPE_RULE}: fração em coluna inteira",
                "check_number": None,
                "failure_case": value,
                "index": index,
            }
            for index, value in fractional.items()
        )
    return pd.DataFrame(rows, columns=FAILURE_COLUMNS)


def validate_batch(frame: pd.DataFrame, contract: pa.DataFrameSchema, batch: str) -> ContractResult:
    """Valida ``frame`` e devolve o resultado, sem levantar excecao e sem alterar o lote.

    Limite conhecido: quando uma coluna numerica traz texto, o Pandera nao consegue avaliar as
    demais regras daquela coluna (faixa, dominio); so a violacao de tipo (DQ-10) e reportada, e as
    outras aparecem depois que o tipo for corrigido na origem.
    """
    result = ContractResult(batch=batch, contract=str(contract.name), n_rows=len(frame), n_columns=frame.shape[1])
    cases = _fractional_cases(frame, contract)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            contract.validate(frame, lazy=True)
        except SchemaErrors as exc:
            cases = pd.concat([exc.failure_cases[FAILURE_COLUMNS], cases], ignore_index=True)
    if not cases.empty:
        result.violations, result.rows_with_failures = _summarize(cases, set(contract.columns))
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


def quarantine_batch(
    frame: pd.DataFrame, result: ContractResult, quarantine_dir: Path, contract_columns: set[str] | None = None
) -> Path:
    """Grava o lote reprovado e o seu relatorio na quarentena (acesso restrito, retencao curta).

    Com ``contract_columns``, as colunas fora do contrato **nao sao gravadas**: se a origem enviou
    um dado pessoal por engano (ex.: ``cpf``), a quarentena nao pode se tornar uma copia dele. O
    nome da coluna descartada fica no relatorio (regra DQ-09), o que basta para a origem corrigir.
    """
    target = Path(quarantine_dir) / result.batch
    target.mkdir(parents=True, exist_ok=True)
    if contract_columns is not None:
        frame = frame[[column for column in frame.columns if column in contract_columns]]
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
            location = quarantine_batch(frame, result, quarantine_dir, set(contract.columns))
            logger.error("lote %s BLOQUEADO; copia em quarentena em %s", batch, location)
        raise DataContractError(result)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SchemaWarning)
        validated = contract.validate(frame)
    return validated, result
