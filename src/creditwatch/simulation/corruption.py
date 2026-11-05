"""Injecao controlada de defeitos em um lote valido, para demonstrar o contrato bloqueando a ingestao.

Cada injetor reproduz um incidente tipico de pipeline de dados e viola exatamente uma regra do
contrato, o que permite testar as regras uma a uma e montar o "lote com erros" da demonstracao.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from creditwatch.data.schema import BILL_AMT, LIMIT, PAY_AMT, PAY_STATUS

Injector = Callable[[pd.DataFrame, np.random.Generator], pd.DataFrame]


def _rows(frame: pd.DataFrame, rng: np.random.Generator, fraction: float) -> np.ndarray:
    size = max(1, int(len(frame) * fraction))
    return rng.choice(frame.index.to_numpy(), size=size, replace=False)


def null_limits(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Falha de integracao: limite de credito chega nulo em 3% das linhas (DQ-01)."""
    out = frame.copy()
    out[LIMIT] = out[LIMIT].astype("float64")
    out.loc[_rows(out, rng, 0.03), LIMIT] = np.nan
    return out


def underage_and_impossible_ages(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Menores de idade e idades impossiveis, tipicas de erro de digitacao ou de formulario (DQ-02)."""
    out = frame.copy()
    rows = _rows(out, rng, 0.02)
    half = len(rows) // 2
    out.loc[rows[:half], "age"] = rng.integers(12, 18, size=half)
    out.loc[rows[half:], "age"] = rng.integers(120, 200, size=len(rows) - half)
    return out


def limits_above_policy(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Limite em centavos em vez de unidades monetarias: estoura o teto da politica (DQ-03)."""
    out = frame.copy()
    rows = _rows(out, rng, 0.01)
    out.loc[rows, LIMIT] = out.loc[rows, LIMIT] * 100
    return out


def duplicated_customers(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Reenvio parcial do arquivo: 2% das propostas aparecem duas vezes (DQ-04)."""
    return pd.concat([frame, frame.loc[_rows(frame, rng, 0.02)]], ignore_index=True)


def unknown_categories(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Codigos fora do dicionario de dados em sex e education (DQ-05)."""
    out = frame.copy()
    out.loc[_rows(out, rng, 0.01), "sex"] = 0
    out.loc[_rows(out, rng, 0.01), "education"] = 9
    return out


def invalid_pay_status(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Status de pagamento fora da escala -2..8 (DQ-06)."""
    out = frame.copy()
    out.loc[_rows(out, rng, 0.01), PAY_STATUS[0]] = 15
    return out


def negative_payments(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Estornos lancados como pagamento negativo (DQ-07)."""
    out = frame.copy()
    rows = _rows(out, rng, 0.01)
    out.loc[rows, PAY_AMT[0]] = -out.loc[rows, PAY_AMT[0]].abs() - 1
    return out


def absurd_bills(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Fatura com erro de escala (DQ-08)."""
    out = frame.copy()
    out.loc[_rows(out, rng, 0.005), BILL_AMT[0]] = 999_999_999
    return out


def cpf_check_digits(base: str) -> str:
    """Digitos verificadores de um CPF a partir dos nove primeiros digitos."""
    digits = [int(char) for char in base]
    for _ in range(2):
        weights = range(len(digits) + 1, 1, -1)
        remainder = sum(digit * weight for digit, weight in zip(digits, weights, strict=True)) % 11
        digits.append(0 if remainder < 2 else 11 - remainder)
    return f"{digits[-2]}{digits[-1]}"


def fake_cpf(rng: np.random.Generator) -> str:
    """CPF ficticio e garantidamente *invalido*: o ultimo digito verificador e deslocado em uma unidade.

    Assim a demonstracao nunca gera, nem por acaso, o CPF de uma pessoa real.
    """
    base = f"{int(rng.integers(0, 10**9)):09d}"
    check = cpf_check_digits(base)
    return f"{base}{check[0]}{(int(check[1]) + 1) % 10}"


def leaked_pii_column(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """O sistema de origem passa a enviar o CPF do cliente, que nao esta no contrato (DQ-09).

    Os valores sao ficticios e invalidos por construcao (ver ``fake_cpf``).
    """
    out = frame.copy()
    out["cpf"] = [fake_cpf(rng) for _ in range(len(out))]
    return out


def missing_column(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Mudanca de layout na origem: uma coluna obrigatoria desaparece (DQ-09)."""
    return frame.drop(columns=[PAY_AMT[-1]])


def text_in_numeric_column(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Texto em coluna numerica (DQ-10)."""
    out = frame.copy()
    out["age"] = out["age"].astype("object")
    out.loc[_rows(out, rng, 0.005), "age"] = "N/A"
    return out


def truncated_batch(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Arquivo truncado na transferencia: restam poucas linhas (DQ-11)."""
    return frame.head(50)


INJECTORS: dict[str, tuple[str, Injector]] = {
    "null_limits": ("DQ-01", null_limits),
    "invalid_ages": ("DQ-02", underage_and_impossible_ages),
    "limits_above_policy": ("DQ-03", limits_above_policy),
    "duplicated_customers": ("DQ-04", duplicated_customers),
    "unknown_categories": ("DQ-05", unknown_categories),
    "invalid_pay_status": ("DQ-06", invalid_pay_status),
    "negative_payments": ("DQ-07", negative_payments),
    "absurd_bills": ("DQ-08", absurd_bills),
    "leaked_pii_column": ("DQ-09", leaked_pii_column),
    "missing_column": ("DQ-09", missing_column),
    "text_in_numeric_column": ("DQ-10", text_in_numeric_column),
    "truncated_batch": ("DQ-11", truncated_batch),
}

# Defeitos do lote usado na demonstracao: cobrem as regras exigidas no enunciado (nulos, faixa de
# idade, limites, duplicatas) e a regra de privacidade (coluna de PII fora do contrato).
DEMO_DEFECTS = (
    "null_limits",
    "invalid_ages",
    "limits_above_policy",
    "duplicated_customers",
    "unknown_categories",
    "negative_payments",
    "leaked_pii_column",
)


def corrupt_batch(frame: pd.DataFrame, defects: tuple[str, ...] = DEMO_DEFECTS, seed: int = 42) -> pd.DataFrame:
    """Aplica os defeitos em sequencia e devolve o lote corrompido (o original nao e alterado)."""
    unknown = set(defects) - set(INJECTORS)
    if unknown:
        raise ValueError(f"defeitos desconhecidos: {sorted(unknown)}")
    rng = np.random.default_rng(seed)
    out = frame.copy()
    for name in defects:
        out = INJECTORS[name][1](out, rng)
    return out


def expected_rules(defects: tuple[str, ...] = DEMO_DEFECTS) -> list[str]:
    """Regras que o contrato deve acusar para um conjunto de defeitos."""
    return sorted({INJECTORS[name][0] for name in defects})
