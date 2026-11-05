"""Contrato de dados (Pandera) dos lotes de credito.

O mesmo contrato protege o treino (Dataset de Referencia, com rotulo) e a escoragem (lotes de
producao, sem rotulo). Ele e ``strict``: qualquer coluna fora do contrato reprova o lote, o que
implementa minimizacao de dados por padrao (um ``cpf`` enviado por engano nunca chega ao modelo).

Regras rigidas tratam de valores *invalidos*; mudancas de *distribuicao* de valores validos nao
sao assunto do contrato e sim da deteccao de drift.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

from creditwatch.contracts.rules import RULES_BY_ID
from creditwatch.data.schema import (
    BATCH_MONTH,
    BILL_AMT,
    EDUCATION_DOMAIN,
    ID_COLUMN,
    LIMIT,
    MARRIAGE_DOMAIN,
    PAY_AMT,
    PAY_STATUS,
    PAY_STATUS_MAX,
    PAY_STATUS_MIN,
    SEX_DOMAIN,
    TARGET,
)

AGE_MIN, AGE_MAX = 18, 100
LIMIT_MIN, LIMIT_MAX = 1_000, 1_000_000
BILL_MIN, BILL_MAX = -1_000_000, 10_000_000
MAX_INACTIVE_SHARE = 0.60
INTEGER = "int64"


def _error(rule_id: str) -> str:
    """Mensagem do check: comeca pelo identificador para o relatorio agrupar as falhas por regra."""
    return f"{rule_id}: {RULES_BY_ID[rule_id].title}"


def _min_rows(min_rows: int):
    return pa.Check(lambda frame: len(frame) >= min_rows, error=_error("DQ-11"))


def _single_month(frame: pd.DataFrame) -> bool:
    return BATCH_MONTH not in frame.columns or frame[BATCH_MONTH].nunique(dropna=False) == 1


def _inactive_share(frame: pd.DataFrame) -> bool:
    if PAY_STATUS[0] not in frame.columns:
        return True  # a ausencia da coluna ja e reportada pela regra de esquema (DQ-09)
    status = pd.to_numeric(frame[PAY_STATUS[0]], errors="coerce")
    return bool((status == PAY_STATUS_MIN).mean() < MAX_INACTIVE_SHARE)


def build_contract(require_target: bool, min_rows: int = 500) -> pa.DataFrameSchema:
    """Monta o contrato de um lote.

    ``require_target=True`` para o Dataset de Referencia (o rotulo e obrigatorio); ``False`` para
    lotes de producao no momento da escoragem, quando o rotulo ainda nao existe e, por ser o
    contrato estrito, nem pode estar presente.
    """
    columns: dict[str, pa.Column] = {
        ID_COLUMN: pa.Column(INTEGER, unique=True, description="Identificador da proposta."),
        LIMIT: pa.Column(INTEGER, pa.Check.in_range(LIMIT_MIN, LIMIT_MAX, error=_error("DQ-03"))),
        "sex": pa.Column(INTEGER, pa.Check.isin(list(SEX_DOMAIN), error=_error("DQ-05"))),
        "education": pa.Column(INTEGER, pa.Check.isin(list(EDUCATION_DOMAIN), error=_error("DQ-05"))),
        "marriage": pa.Column(INTEGER, pa.Check.isin(list(MARRIAGE_DOMAIN), error=_error("DQ-05"))),
        "age": pa.Column(INTEGER, pa.Check.in_range(AGE_MIN, AGE_MAX, error=_error("DQ-02"))),
        BATCH_MONTH: pa.Column(
            str, pa.Check.str_matches(r"^\d{4}-(0[1-9]|1[0-2])$", error=_error("DQ-13")), required=False
        ),
    }
    for column in PAY_STATUS:
        columns[column] = pa.Column(INTEGER, pa.Check.in_range(PAY_STATUS_MIN, PAY_STATUS_MAX, error=_error("DQ-06")))
    for column in BILL_AMT:
        columns[column] = pa.Column(INTEGER, pa.Check.in_range(BILL_MIN, BILL_MAX, error=_error("DQ-08")))
    for column in PAY_AMT:
        columns[column] = pa.Column(INTEGER, pa.Check.ge(0, error=_error("DQ-07")))
    if require_target:
        columns[TARGET] = pa.Column(INTEGER, pa.Check.isin([0, 1], error=_error("DQ-12")))

    return pa.DataFrameSchema(
        columns,
        checks=[
            _min_rows(min_rows),
            pa.Check(_single_month, error=_error("DQ-13")),
            pa.Check(_inactive_share, error=_error("DQ-W01"), raise_warning=True),
        ],
        strict=True,
        coerce=True,
        name="credit_batch_with_target" if require_target else "credit_batch",
        description="Contrato de dados dos lotes de credito (creditwatch).",
    )


def build_label_contract() -> pa.DataFrameSchema:
    """Contrato do arquivo de rotulos que chega apos a maturacao do lote."""
    return pa.DataFrameSchema(
        {
            ID_COLUMN: pa.Column(INTEGER, unique=True),
            TARGET: pa.Column(INTEGER, pa.Check.isin([0, 1], error=_error("DQ-12"))),
        },
        strict=True,
        coerce=True,
        name="credit_labels",
    )
