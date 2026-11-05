"""Engenharia de features compartilhada por treino, escoragem e monitoramento.

Uma unica funcao pura garante que o modelo veja em producao exatamente as mesmas transformacoes do
treino (sem *training-serving skew*). As razoes (utilizacao do limite, razao de pagamento) sao
adimensionais: a razao de pagamento nao muda se fatura e pagamento forem inflacionados pelo mesmo
fator, enquanto a utilizacao sobe quando a fatura cresce e o limite fica parado (sinal de aperto).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from creditwatch.data.schema import BILL_AMT, ENGINEERED, LIMIT, MODEL_FEATURES, PAY_AMT, PAY_STATUS

UTILIZATION_BOUNDS = (-1.0, 5.0)
PAYMENT_RATIO_BOUNDS = (0.0, 2.0)
TREND_BOUNDS = (-5.0, 5.0)


def _payment_ratio(paid: pd.Series | np.ndarray, billed: pd.Series | np.ndarray) -> np.ndarray:
    """Fracao da fatura paga; sem saldo devedor conta como pagamento integral (1.0)."""
    paid = np.asarray(paid, dtype=float)
    billed = np.asarray(billed, dtype=float)
    ratio = np.divide(paid, billed, out=np.ones_like(paid), where=billed > 0)
    return np.clip(ratio, *PAYMENT_RATIO_BOUNDS)


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Devolve uma copia de ``frame`` com as colunas de ``ENGINEERED`` adicionadas."""
    out = frame.copy()
    limit = out[LIMIT].astype(float)
    bills = out[list(BILL_AMT)].astype(float)
    pays = out[list(PAY_AMT)].astype(float)
    status = out[list(PAY_STATUS)]

    out["utilization_1"] = (bills[BILL_AMT[0]] / limit).clip(*UTILIZATION_BOUNDS)
    out["utilization_avg"] = (bills.mean(axis=1) / limit).clip(*UTILIZATION_BOUNDS)
    # O pagamento do mes m quita a fatura do mes m+1 (um mes mais antiga).
    out["payment_ratio_1"] = _payment_ratio(pays[PAY_AMT[0]], bills[BILL_AMT[1]])
    billed_total = bills[list(BILL_AMT[1:])].clip(lower=0).sum(axis=1)
    out["payment_ratio_avg"] = _payment_ratio(pays[list(PAY_AMT[:-1])].sum(axis=1), billed_total)
    out["max_delay"] = status.max(axis=1).clip(lower=0).astype("int64")
    out["months_delayed"] = (status >= 1).sum(axis=1).astype("int64")
    out["bill_trend"] = ((bills[BILL_AMT[0]] - bills[BILL_AMT[-1]]) / limit).clip(*TREND_BOUNDS)
    return out


def model_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Matriz de entrada do modelo, na ordem fixa de ``MODEL_FEATURES``."""
    missing = [c for c in ENGINEERED if c not in frame.columns]
    enriched = add_engineered_features(frame) if missing else frame
    return enriched[list(MODEL_FEATURES)].astype(float)
