"""Mecanismos de degradacao aplicados aos lotes de producao.

Cada mecanismo e uma intervencao isolada e parametrizada por uma intensidade, o que permite
liga-los e desliga-los um a um na analise causal (``creditwatch.analysis.causal``):

- ``new_profiles`` (drift de dados): a carteira passa a receber clientes mais jovens e com limite
  menor. E uma reamostragem do pool: muda a composicao da carteira e nao altera o rotulo de nenhum
  cliente. Como a idade nao e entrada do modelo, a relacao entre as *features do modelo* e a
  inadimplencia pode variar um pouco com a composicao.
- ``stress`` (drift de dados): cresce a parcela de clientes com atraso recente (aperto
  financeiro). Tambem e reamostragem, sobre variaveis que o modelo enxerga.
- ``inflation`` (drift de dados): as faturas sobem em termos nominais e os pagamentos nao
  acompanham. O rotulo nao muda; por isso, medida em valores nominais, a relacao entre features e
  inadimplencia se desloca (o mesmo risco aparece com faturas maiores).
- ``concept`` (drift de conceito): bons pagadores alavancados passam a inadimplir e parte dos
  inadimplentes e curada por renegociacao. Nenhuma feature muda, so o rotulo.
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
    """Pesos de amostragem (*exponential tilting*) que mudam a composicao da carteira.

    ``new_profiles`` favorece clientes jovens e de limite baixo; ``stress`` favorece clientes com
    atraso recente. Com as duas intensidades em zero a amostragem e uniforme.
    """
    profile_score = -(_zscore(pool["age"]) + _zscore(np.log(pool[LIMIT]))) / 2
    recent_delay = pool[list(PAY_STATUS[:2])].clip(lower=0).sum(axis=1)
    stress_score = _zscore(recent_delay)
    weights = np.exp(new_profiles * profile_score + stress * stress_score)
    return weights / weights.sum()


def sample_positions(
    pool: pd.DataFrame, n: int, rng: np.random.Generator, new_profiles: float = 0.0, stress: float = 0.0
) -> np.ndarray:
    """Posicoes (no pool) de ``n`` clientes sorteados sem reposicao, com os pesos de ``sampling_weights``.

    Usa o truque de Gumbel: somar um ruido de Gumbel ao log do peso de cada cliente e ficar com os
    ``n`` maiores equivale a sortear sem reposicao com probabilidade proporcional ao peso. O ruido
    e sorteado uma vez para o pool inteiro; assim, com a mesma seed, dois cenarios que diferem so
    nos pesos compartilham o ruido (numeros aleatorios comuns) e as amostras mudam apenas o quanto
    os pesos mandam. Isso reduz a variancia das comparacoes entre cenarios.
    """
    if n > len(pool):
        raise ValueError(f"lote de {n} propostas maior que o pool de {len(pool)} clientes")
    keys = np.log(sampling_weights(pool, new_profiles, stress)) + rng.gumbel(size=len(pool))
    top = np.argpartition(-keys, n - 1)[:n]
    return top[np.argsort(-keys[top], kind="stable")]


def sample_batch(
    pool: pd.DataFrame, n: int, rng: np.random.Generator, new_profiles: float = 0.0, stress: float = 0.0
) -> pd.DataFrame:
    """Amostra ``n`` clientes do pool (sem reposicao) com os pesos de ``sampling_weights``."""
    return pool.iloc[sample_positions(pool, n, rng, new_profiles, stress)].reset_index(drop=True)


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


def apply_concept_drift(
    batch: pd.DataFrame, intensity: float, rng: np.random.Generator | None = None, draws: np.ndarray | None = None
) -> pd.DataFrame:
    """Altera a relacao entre features e inadimplencia, sem mexer nas features.

    - Clientes em dia e alavancados (utilizacao >= 50%) passam a inadimplir com probabilidade
      proporcional a ``intensity``: o sinal em que o modelo mais confia deixa de proteger.
    - Inadimplentes com atraso recente sao curados por programas de renegociacao.

    ``draws`` sao os numeros uniformes (um por linha) que decidem quem muda de rotulo. Informa-los
    permite que o mesmo cliente tenha o mesmo sorteio em cenarios diferentes; sem eles, sao
    sorteados de ``rng``.
    """
    if intensity <= 0:
        return batch
    if draws is None:
        if rng is None:
            raise ValueError("informe rng ou draws")
        draws = rng.random(len(batch))
    out = batch.copy()
    utilization = out[BILL_AMT[0]] / out[LIMIT]
    on_time = out[PAY_STATUS[0]] <= 0
    leveraged_good = on_time & (utilization >= LEVERAGED_UTILIZATION) & (out[TARGET] == 0)
    delinquent_bad = (out[PAY_STATUS[0]] >= 1) & (out[TARGET] == 1)

    draws = np.asarray(draws)
    new_defaults = leveraged_good & (draws < intensity * MAX_NEW_DEFAULT_PROBABILITY)
    cured = delinquent_bad & (draws < intensity * MAX_CURE_PROBABILITY)
    out.loc[new_defaults, TARGET] = 1
    out.loc[cured, TARGET] = 0
    return out
