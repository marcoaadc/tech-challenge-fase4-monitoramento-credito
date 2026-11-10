"""Experimento de remediacao: o que recupera o modelo depois do drift?

Simula a decisao que a equipe tomaria antes dos meses de teste, usando apenas rotulos que ja
estariam maduros naquela data, e compara estrategias em meses posteriores (*out-of-time*):

- ``campeao``: manter o modelo atual (nao fazer nada);
- ``campeao_recalibrado_por_segmento``: manter o modelo e corrigir a PD de cada segmento de
  comportamento pela razao entre inadimplencia observada e prevista nas safras recentes;
- ``retreino_janela_recente``: retreinar so com os meses de producao mais recentes;
- ``retreino_referencia_mais_producao``: retreinar com a referencia mais toda a producao rotulada.

Tres cuidados tornam a comparacao justa:

1. **sem vazamento temporal**: todo mes de treino precisa ter o rotulo maduro antes do primeiro
   mes de teste (``label_lag_months``);
2. **sem vazamento de cliente**: nenhum cliente de origem aparece no treino e no teste, e o ponto
   de corte de cada desafiante e escolhido em clientes que ele nao viu (``group_column``);
3. **mesmo volume de aprovacao**: a inadimplencia entre aprovados tambem e medida aprovando a
   mesma fracao de propostas em todas as estrategias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import GroupShuffleSplit

from creditwatch.analysis.causal import assign_segment
from creditwatch.data.schema import BATCH_MONTH, SCORE, TARGET
from creditwatch.features.engineering import add_engineered_features, model_matrix
from creditwatch.model.metrics import best_ks_threshold, classification_metrics
from creditwatch.model.training import build_candidate, predict_default_probability
from creditwatch.observability.alerts import add_months

REPORTED = (
    "roc_auc",
    "ks",
    "brier",
    "calibration_gap",
    "approval_rate",
    "bad_rate_approved",
    "bad_rate_at_champion_approval",
)
STRATEGIES = (
    "campeao",
    "campeao_recalibrado_por_segmento",
    "retreino_janela_recente",
    "retreino_referencia_mais_producao",
)
CEILING = "teto_rotulos_imediatos"
VALIDATION_FRACTION = 0.2
MIN_SEGMENT_ROWS = 100


def _fit_challenger(train: pd.DataFrame, groups: np.ndarray, model_type: str, seed: int) -> tuple[BaseEstimator, float]:
    """Treina um desafiante e escolhe o ponto de corte em uma fatia de validacao.

    A fatia e separada por grupo (cliente de origem): o mesmo cliente pode aparecer em varios
    meses de treino, e o corte nao pode ser escolhido em clientes que o modelo ja viu.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=VALIDATION_FRACTION, random_state=seed)
    fit_index, validation_index = next(splitter.split(train, groups=groups))
    fit_rows, validation = train.iloc[fit_index], train.iloc[validation_index]
    model = build_candidate(model_type, seed)
    model.fit(model_matrix(fit_rows), fit_rows[TARGET])
    threshold = best_ks_threshold(validation[TARGET].to_numpy(), predict_default_probability(model, validation))
    return model, threshold


def segment_calibration_factors(recent: pd.DataFrame, score: np.ndarray, cut: float) -> dict[str, float]:
    """Razao entre inadimplencia observada e PD prevista em cada segmento das safras recentes.

    Um fator acima de 1 indica um segmento em que o modelo subestima o risco. Segmentos com menos
    de ``MIN_SEGMENT_ROWS`` linhas ficam com fator 1 (sem evidencia suficiente para corrigir).
    """
    frame = pd.DataFrame(
        {"segment": assign_segment(recent, cut).to_numpy(), TARGET: recent[TARGET].to_numpy(), SCORE: score}
    )
    factors: dict[str, float] = {}
    for segment, part in frame.groupby("segment"):
        predicted = part[SCORE].mean()
        enough = len(part) >= MIN_SEGMENT_ROWS and predicted > 0
        factors[str(segment)] = float(part[TARGET].mean() / predicted) if enough else 1.0
    return factors


def recalibrate_by_segment(frame: pd.DataFrame, score: np.ndarray, factors: dict[str, float], cut: float) -> np.ndarray:
    """Aplica os fatores de ``segment_calibration_factors`` a PD de cada proposta."""
    multiplier = assign_segment(frame, cut).map(factors).fillna(1.0).to_numpy()
    return np.clip(np.asarray(score) * multiplier, 0.0, 1.0)


def bad_rate_at_approval(y_true: np.ndarray, score: np.ndarray, approval_rate: float) -> float:
    """Inadimplencia entre os aprovados quando se aprova a fracao ``approval_rate`` de menor score.

    Cada estrategia tem o seu ponto de corte, entao as taxas de aprovacao diferem e a inadimplencia
    entre aprovados deixa de ser comparavel. Fixar o volume aprovado isola a qualidade da ordenacao.
    """
    n_approved = round(approval_rate * len(score))
    if n_approved == 0:
        return float("nan")
    approved = np.argsort(score, kind="stable")[:n_approved]
    return float(np.asarray(y_true)[approved].mean())


def _evaluate(y_true: np.ndarray, score: np.ndarray, threshold: float, approval_rate: float) -> dict[str, float]:
    metrics = classification_metrics(y_true, score, threshold)
    metrics["bad_rate_at_champion_approval"] = bad_rate_at_approval(y_true, score, approval_rate)
    return {key: round(metrics[key], 4) for key in REPORTED}


def _months(frame: pd.DataFrame) -> list[str]:
    return sorted(frame[BATCH_MONTH].unique())


def remediation_experiment(
    champion: BaseEstimator,
    champion_threshold: float,
    model_type: str,
    reference_train: pd.DataFrame,
    train_production: pd.DataFrame,
    test_production: pd.DataFrame,
    recent_months: list[str],
    segment_cut: float,
    seed: int = 42,
    group_column: str | None = None,
    label_lag_months: int = 0,
    oracle_production: pd.DataFrame | None = None,
) -> dict:
    """Compara as estrategias em ``test_production``.

    ``train_production`` traz os meses de producao com rotulo maduro na data da decisao e
    ``recent_months`` e o subconjunto mais recente deles (janela deslizante). ``segment_cut`` e o
    corte de utilizacao usado na segmentacao (``analysis.causal.utilization_cut``).

    ``oracle_production`` sao lotes dos proprios meses de teste, de outros clientes. Um modelo
    treinado neles (``CEILING``) mostra o teto: o melhor que um retreino conseguiria se o rotulo
    fosse imediato. Nao e uma estrategia aplicavel, e uma referencia para ler as demais.
    """
    train_months, test_months = _months(train_production), _months(test_production)
    if set(train_months) & set(test_months):
        raise ValueError("meses de teste nao podem estar no treino")
    if add_months(train_months[-1], label_lag_months) > test_months[0]:
        raise ValueError(
            f"o rotulo de {train_months[-1]} so amadurece em {add_months(train_months[-1], label_lag_months)}, "
            f"depois do primeiro mes de teste ({test_months[0]})"
        )
    if group_column is not None:
        shared = set(train_production[group_column]) & set(test_production[group_column])
        if shared:
            raise ValueError(f"{len(shared)} clientes aparecem no treino e no teste")

    # As features derivadas sao (re)calculadas aqui: misturar um frame sem elas com outro que as tem
    # geraria colunas nulas ao concatenar referencia e producao.
    reference_train = add_engineered_features(reference_train).reset_index(drop=True)
    labeled = add_engineered_features(train_production).reset_index(drop=True)
    test = add_engineered_features(test_production).reset_index(drop=True)
    recent = labeled[labeled[BATCH_MONTH].isin(recent_months)].reset_index(drop=True)
    if recent.empty:
        raise ValueError("nenhum mes de treino dentro da janela recente")

    def groups_of(frame: pd.DataFrame) -> np.ndarray:
        if group_column is None:
            return np.arange(len(frame)).astype(str)
        return frame[group_column].astype(str).to_numpy()

    reference_groups = np.char.add("ref-", np.arange(len(reference_train)).astype(str))
    combined = pd.concat([reference_train, labeled], ignore_index=True)
    combined_groups = np.concatenate([reference_groups, groups_of(labeled)])

    y_test = test[TARGET].to_numpy()
    champion_score = predict_default_probability(champion, test)
    champion_approval = float((champion_score < champion_threshold).mean())
    factors = segment_calibration_factors(recent, predict_default_probability(champion, recent), segment_cut)
    recent_model, recent_threshold = _fit_challenger(recent, groups_of(recent), model_type, seed)
    combined_model, combined_threshold = _fit_challenger(combined, combined_groups, model_type, seed)

    # nome -> (score no teste, ponto de corte, linhas usadas para ajustar a estrategia)
    strategies = {
        "campeao": (champion_score, champion_threshold, len(reference_train)),
        "campeao_recalibrado_por_segmento": (
            recalibrate_by_segment(test, champion_score, factors, segment_cut),
            champion_threshold,
            len(recent),
        ),
        "retreino_janela_recente": (predict_default_probability(recent_model, test), recent_threshold, len(recent)),
        "retreino_referencia_mais_producao": (
            predict_default_probability(combined_model, test),
            combined_threshold,
            len(combined),
        ),
    }
    if oracle_production is not None:
        oracle = add_engineered_features(oracle_production).reset_index(drop=True)
        if group_column is not None and set(oracle[group_column]) & set(test[group_column]):
            raise ValueError("clientes do teto aparecem no teste")
        oracle_model, oracle_threshold = _fit_challenger(oracle, groups_of(oracle), model_type, seed)
        strategies[CEILING] = (predict_default_probability(oracle_model, test), oracle_threshold, len(oracle))
    return {
        "model_type": model_type,
        "train_months": train_months,
        "recent_months": [month for month in train_months if month in set(recent_months)],
        "test_months": test_months,
        "label_lag_months": label_lag_months,
        "test_rows": len(test),
        "disjoint_customers": group_column is not None,
        "champion_approval_rate": round(champion_approval, 4),
        "segment_calibration_factors": {segment: round(factor, 3) for segment, factor in factors.items()},
        "strategies": {
            name: {
                "train_rows": int(rows),
                "threshold": round(float(threshold), 4),
                **_evaluate(y_test, score, threshold, champion_approval),
            }
            for name, (score, threshold, rows) in strategies.items()
        },
    }


def average_folds(folds: list[dict]) -> dict:
    """Media das metricas de cada estrategia entre as dobras, guardando o ROC AUC de cada dobra."""
    first = folds[0]
    strategies = {}
    for name in first["strategies"]:
        rows = [fold["strategies"][name] for fold in folds]
        strategies[name] = {
            key: round(float(np.mean([row[key] for row in rows])), 4) for key in ("train_rows", "threshold", *REPORTED)
        }
        strategies[name]["train_rows"] = round(strategies[name]["train_rows"])
        strategies[name]["roc_auc_by_fold"] = [row["roc_auc"] for row in rows]
    segments = first["segment_calibration_factors"]
    return {
        **{key: first[key] for key in ("model_type", "train_months", "recent_months", "test_months")},
        "label_lag_months": first["label_lag_months"],
        "disjoint_customers": all(fold["disjoint_customers"] for fold in folds),
        "folds": len(folds),
        "test_rows": int(sum(fold["test_rows"] for fold in folds)),
        "champion_approval_rate": round(float(np.mean([fold["champion_approval_rate"] for fold in folds])), 4),
        "segment_calibration_factors": {
            segment: round(float(np.mean([fold["segment_calibration_factors"].get(segment, 1.0) for fold in folds])), 3)
            for segment in segments
        },
        "strategies": strategies,
    }
