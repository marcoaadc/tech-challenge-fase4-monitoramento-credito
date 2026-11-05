"""Mecanismos de degradacao aplicados aos lotes de producao.

Cada mecanismo e uma intervencao isolada e parametrizada por uma intensidade, o que permite
liga-los e desliga-los um a um na analise causal (``creditwatch.analysis.causal``):

- ``new_profiles`` (drift de dados): muda P(x). A carteira passa a receber clientes mais jovens e
  com limite menor. Implementado por reamostragem, logo P(y|x) e preservada.
- ``stress`` (drift de dados): muda P(x). Cresce a parcela de clientes com atraso recente
  (aperto financeiro). Tambem por reamostragem, com P(y|x) preservada.
- ``inflation`` (drift de dados): muda x. As faturas sobem em termos nominais e os pagamentos nao
  acompanham; o risco real do cliente e o rotulo nao mudam.
- ``concept`` (drift de conceito): muda P(y|x). Bons pagadores alavancados passam a inadimplir e
  parte dos inadimplentes e curada por renegociacao; as features nao mudam.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from creditwatch.data.schema import BILL_AMT, LIMIT, PAY_AMT, PAY_STATUS, TARGET

# Quanto do aumento nominal das faturas e acompanhado pelos pagamentos (renda defasada).
PAYMENT_PASS_THROUGH = 0.35
# Segmento afetado pelo drift de conceito: em dia e com utilizacao do limite a partir deste valor.
LEVERAGED_UTILIZATION = 0.5
MAX_NEW_DEFAULT_PROBABILITY = 0.40
MAX_CURE_PROBABILITY = 0.30


def _zscore(values: pd.Series) -> np.ndarray:
    values = values.astype(float)
    std = values.std(ddof=0)
    return ((values - values.mean()) / std).to_numpy() if std > 0 else np.zeros(len(values))


def sampling_weights(pool: pd.DataFrame, new_profiles: float = 0.0, stress: float = 0.0) -> np.ndarray:
    """Pesos de amostragem (*exponential tilting*) que deslocam P(x) sem tocar em P(y|x).

    ``new_profiles`` favorece clientes jovens e de limite baixo; ``stress`` favorece clientes com
    atraso recente. Com as duas intensidades em zero a amostragem e uniforme.
    """
    profile_score = -(_zscore(pool["age"]) + _zscore(np.log(pool[LIMIT]))) / 2
    recent_delay = pool[list(PAY_STATUS[:2])].clip(lower=0).sum(axis=1)
    stress_score = _zscore(recent_delay)
    weights = np.exp(new_profiles * profile_score + stress * stress_score)
    return weights / weights.sum()


def sample_batch(
    pool: pd.DataFrame, n: int, rng: np.random.Generator, new_profiles: float = 0.0, stress: float = 0.0
) -> pd.DataFrame:
    """Amostra ``n`` clientes do pool (sem reposicao) com os pesos de ``sampling_weights``."""
    weights = sampling_weights(pool, new_profiles, stress)
    chosen = rng.choice(len(pool), size=n, replace=False, p=weights)
    return pool.iloc[chosen].reset_index(drop=True)


def apply_inflation(batch: pd.DataFrame, price_index: float) -> pd.DataFrame:
    """Infla as faturas em ``price_index`` (0,30 = +30%) e os pagamentos em apenas uma fracao disso.

    O historico e inflado de forma gradual: o mes mais recente recebe o indice cheio e o mais
    antigo um sexto dele. O limite de credito e o rotulo permanecem iguais.
    """
    if price_index <= 0:
        return batch
    out = batch.copy()
    n_months = len(BILL_AMT)
    for position, (bill, pay) in enumerate(zip(BILL_AMT, PAY_AMT, strict=True)):
        month_index = price_index * (n_months - position) / n_months
        out[bill] = (out[bill] * (1 + month_index)).round().astype("int64")
        out[pay] = (out[pay] * (1 + PAYMENT_PASS_THROUGH * month_index)).round().astype("int64")
    return out


def apply_concept_drift(batch: pd.DataFrame, intensity: float, rng: np.random.Generator) -> pd.DataFrame:
    """Altera a relacao entre features e inadimplencia, sem mexer nas features.

    - Clientes em dia e alavancados (utilizacao >= 50%) passam a inadimplir com probabilidade
      proporcional a ``intensity``: o sinal em que o modelo mais confia deixa de proteger.
    - Inadimplentes com atraso recente sao curados por programas de renegociacao.
    """
    if intensity <= 0:
        return batch
    out = batch.copy()
    utilization = out[BILL_AMT[0]] / out[LIMIT]
    on_time = out[PAY_STATUS[0]] <= 0
    leveraged_good = on_time & (utilization >= LEVERAGED_UTILIZATION) & (out[TARGET] == 0)
    delinquent_bad = (out[PAY_STATUS[0]] >= 1) & (out[TARGET] == 1)

    draws = rng.random(len(out))
    new_defaults = leveraged_good & (draws < intensity * MAX_NEW_DEFAULT_PROBABILITY)
    cured = delinquent_bad & (draws < intensity * MAX_CURE_PROBABILITY)
    out.loc[new_defaults, TARGET] = 1
    out.loc[cured, TARGET] = 0
    return out
