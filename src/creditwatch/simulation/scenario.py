"""Cenario temporal de producao: doze lotes mensais com degradacao progressiva.

Narrativa: apos o treino (safra de referencia), a economia sofre um choque inflacionario e a
fintech passa a captar um publico mais jovem. Nos tres primeiros meses nada muda (controle de
falsos positivos); de abril a agosto ha drift de dados; a partir de setembro soma-se o drift de
conceito, quando o proprio comportamento de pagamento se altera.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from creditwatch.data.schema import BATCH_MONTH, ID_COLUMN
from creditwatch.simulation.mechanisms import apply_concept_drift, apply_inflation, sample_batch

STABLE, DATA_DRIFT, CONCEPT_DRIFT = "estavel", "drift_de_dados", "drift_de_conceito"
MECHANISMS = ("inflation", "stress", "new_profiles", "concept")


@dataclass(frozen=True)
class MonthPlan:
    """Intensidade de cada mecanismo em um mes de producao."""

    month: str
    phase: str
    inflation: float = 0.0  # indice de precos acumulado desde a referencia (0,30 = +30%)
    stress: float = 0.0  # intensidade do deslocamento para clientes com atraso
    new_profiles: float = 0.0  # intensidade do deslocamento para clientes jovens/limite baixo
    concept: float = 0.0  # intensidade da mudanca em P(y|x), de 0 a 1

    def as_dict(self) -> dict:
        return asdict(self)

    def only(self, *mechanisms: str) -> MonthPlan:
        """Copia do plano mantendo ligados apenas os mecanismos informados (intervencao)."""
        unknown = set(mechanisms) - set(MECHANISMS)
        if unknown:
            raise ValueError(f"mecanismos desconhecidos: {sorted(unknown)}")
        return replace(self, **{name: 0.0 for name in MECHANISMS if name not in mechanisms})


def default_scenario(year: int = 2025) -> list[MonthPlan]:
    """Plano padrao: 3 meses estaveis, 5 de drift de dados e 4 com drift de conceito."""

    def month(number: int) -> str:
        return f"{year}-{number:02d}"

    stable = [MonthPlan(month(m), STABLE) for m in (1, 2, 3)]
    data_drift = [
        MonthPlan(month(4), DATA_DRIFT, inflation=0.10, stress=0.10),
        MonthPlan(month(5), DATA_DRIFT, inflation=0.20, stress=0.20),
        MonthPlan(month(6), DATA_DRIFT, inflation=0.30, stress=0.30, new_profiles=0.25),
        MonthPlan(month(7), DATA_DRIFT, inflation=0.40, stress=0.40, new_profiles=0.50),
        MonthPlan(month(8), DATA_DRIFT, inflation=0.50, stress=0.50, new_profiles=0.75),
    ]
    concept_drift = [
        MonthPlan(month(9), CONCEPT_DRIFT, inflation=0.55, stress=0.50, new_profiles=0.75, concept=0.25),
        MonthPlan(month(10), CONCEPT_DRIFT, inflation=0.60, stress=0.50, new_profiles=0.75, concept=0.50),
        MonthPlan(month(11), CONCEPT_DRIFT, inflation=0.65, stress=0.50, new_profiles=0.75, concept=0.75),
        MonthPlan(month(12), CONCEPT_DRIFT, inflation=0.70, stress=0.50, new_profiles=0.75, concept=1.00),
    ]
    return [*stable, *data_drift, *concept_drift]


def month_seed(seed: int, month: str) -> int:
    """Seed derivada do mes: cada lote e reprodutivel de forma independente dos demais."""
    return seed * 1_000_003 + int(month.replace("-", ""))


def batch_ids(month: str, n: int) -> np.ndarray:
    """IDs de proposta unicos por lote, no formato AAAAMM + sequencial (inteiro de 64 bits)."""
    return int(month.replace("-", "")) * 100_000 + np.arange(1, n + 1, dtype="int64")


def generate_batch(pool: pd.DataFrame, plan: MonthPlan, n: int, seed: int = 42) -> pd.DataFrame:
    """Gera o lote de producao de um mes aplicando os mecanismos na ordem causal.

    1. composicao da carteira (``new_profiles``, ``stress``): quem chega;
    2. ``inflation``: como os valores nominais desses clientes aparecem;
    3. ``concept``: como eles passam a se comportar (rotulo observado depois).
    """
    rng = np.random.default_rng(month_seed(seed, plan.month))
    batch = sample_batch(pool, n, rng, new_profiles=plan.new_profiles, stress=plan.stress)
    batch = apply_inflation(batch, plan.inflation)
    batch = apply_concept_drift(batch, plan.concept, rng)
    batch[ID_COLUMN] = batch_ids(plan.month, len(batch))
    batch.insert(0, BATCH_MONTH, plan.month)
    return batch
