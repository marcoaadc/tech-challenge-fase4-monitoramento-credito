"""Regras de limpeza que transformam o dataset bruto no Dataset de Referencia.

Cada regra e explicita e auditavel: o relatorio devolvido registra quantas linhas cada uma afetou.
O dataset bruto *nao* passa no contrato de dados (codigos fora do dicionario e duplicatas); so o
resultado desta limpeza pode ser usado para treino.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from creditwatch.data.schema import EDUCATION_DOMAIN, ID_COLUMN, MARRIAGE_DOMAIN

EDUCATION_OTHERS = 4
MARRIAGE_OTHERS = 3


@dataclass(frozen=True)
class CleaningReport:
    rows_in: int
    rows_out: int
    education_recoded: int
    marriage_recoded: int
    duplicates_removed: int

    def as_dict(self) -> dict:
        return asdict(self)


def clean_raw_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Aplica as regras de limpeza e devolve ``(dataset_limpo, relatorio)``.

    1. ``education`` fora do dicionario (0, 5, 6) -> 4 ("outros"): a UCI nao documenta esses codigos.
    2. ``marriage`` fora do dicionario (0) -> 3 ("outros").
    3. Clientes repetidos (mesmas features e mesmo rotulo com IDs diferentes) sao removidos,
       mantendo a primeira ocorrencia: duplicatas inflam artificialmente o peso de um perfil.
    """
    cleaned = frame.copy()
    bad_education = ~cleaned["education"].isin(EDUCATION_DOMAIN)
    cleaned.loc[bad_education, "education"] = EDUCATION_OTHERS
    bad_marriage = ~cleaned["marriage"].isin(MARRIAGE_DOMAIN)
    cleaned.loc[bad_marriage, "marriage"] = MARRIAGE_OTHERS

    payload_columns = [c for c in cleaned.columns if c != ID_COLUMN]
    duplicated = cleaned.duplicated(subset=payload_columns, keep="first")
    cleaned = cleaned.loc[~duplicated].reset_index(drop=True)

    report = CleaningReport(
        rows_in=int(len(frame)),
        rows_out=int(len(cleaned)),
        education_recoded=int(bad_education.sum()),
        marriage_recoded=int(bad_marriage.sum()),
        duplicates_removed=int(duplicated.sum()),
    )
    return cleaned, report
