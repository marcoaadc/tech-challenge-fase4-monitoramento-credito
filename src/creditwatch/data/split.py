"""Separacao entre Dataset de Referencia e populacao de Producao.

O Dataset de Referencia e o unico usado para treinar, validar e testar o modelo. A outra metade
(``production pool``) nunca e vista no treino: e dela que o simulador amostra os lotes mensais de
producao. A particao e estratificada pelo rotulo e deterministica por seed.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from creditwatch.data.schema import TARGET

SPLIT_COLUMN = "split"


def split_reference_production(
    frame: pd.DataFrame, reference_fraction: float = 0.5, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide o dataset limpo em ``(referencia, pool_de_producao)``, sem intersecao."""
    reference, pool = train_test_split(
        frame, train_size=reference_fraction, stratify=frame[TARGET], random_state=seed
    )
    return reference.reset_index(drop=True), pool.reset_index(drop=True)


def assign_model_splits(
    reference: pd.DataFrame, val_fraction: float = 0.2, test_fraction: float = 0.2, seed: int = 42
) -> pd.DataFrame:
    """Marca cada linha da referencia como ``train``, ``val`` ou ``test`` (estratificado)."""
    train, holdout = train_test_split(
        reference, test_size=val_fraction + test_fraction, stratify=reference[TARGET], random_state=seed
    )
    val, test = train_test_split(
        holdout,
        test_size=test_fraction / (val_fraction + test_fraction),
        stratify=holdout[TARGET],
        random_state=seed,
    )
    marked = pd.concat(
        [train.assign(**{SPLIT_COLUMN: "train"}), val.assign(**{SPLIT_COLUMN: "val"}), test.assign(**{SPLIT_COLUMN: "test"})]
    )
    return marked.sort_index().reset_index(drop=True)
