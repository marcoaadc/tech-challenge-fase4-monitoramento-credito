"""Estagio 5 - analises: causalidade, vies e remediacao.

Usa os lotes de producao ja validados e os seus rotulos maduros para responder a tres perguntas
de governanca: *o que causou a degradacao*, *quem foi afetado de forma desigual* e *o que a
corrige*. Os resultados vao para ``reports/analysis/`` (JSON e um resumo em Markdown).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from creditwatch.analysis.causal import (
    calibration_scan,
    mechanism_attribution,
    root_cause_ranking,
    segment_diagnosis,
    utilization_cut,
)
from creditwatch.analysis.fairness import fairness_audit, protected_attribute_ablation
from creditwatch.analysis.remediation import average_folds, remediation_experiment
from creditwatch.config.settings import Settings, get_settings
from creditwatch.data.schema import DECISION, ID_COLUMN, MODEL_FEATURES, SCORE
from creditwatch.data.split import SPLIT_COLUMN
from creditwatch.drift.detector import DriftThresholds
from creditwatch.features.engineering import add_engineered_features
from creditwatch.model.registry import CreditScorer
from creditwatch.observability.alerts import add_months
from creditwatch.observability.events import StageTracker
from creditwatch.pipelines._cli import build_parser, read_json, read_table, start_run, write_json
from creditwatch.pipelines.simulate import MANIFEST_FILENAME
from creditwatch.simulation.scenario import SOURCE_ID, MonthPlan, batch_lineage, generate_batch

logger = logging.getLogger("creditwatch.pipelines.analyze")

REFERENCE_PERIOD = "referencia"
PERIOD_LABELS = {
    "referencia": "Referência",
    "estavel": "Estável",
    "drift_de_dados": "Drift de dados",
    "drift_de_conceito": "Drift de conceito",
}
ATTRIBUTE_LABELS = {"sex": "Sexo", "age_band": "Faixa etária", "education": "Escolaridade", "marriage": "Estado civil"}
STRATEGY_LABELS = {
    "campeao": "Manter o modelo atual",
    "campeao_recalibrado_por_segmento": "Recalibrar a PD por segmento",
    "retreino_janela_recente": "Retreinar com a janela recente",
    "retreino_referencia_mais_producao": "Retreinar com referência + produção",
    "teto_rotulos_imediatos": "Teto: retreino com rótulos imediatos (inviável)",
}
RECENT_WINDOW_MONTHS = 3
ATTRIBUTION_SEEDS = 5
MECHANISM_LABELS = {
    "inflation": "Inflação (faturas nominais)",
    "stress": "Aperto financeiro (mais atrasos)",
    "new_profiles": "Novos perfis (jovens, limite baixo)",
    "concept": "Drift de conceito (P(y|x))",
}


def load_labeled_production(settings: Settings, scorer: CreditScorer, manifest: dict) -> pd.DataFrame:
    """Concatena as remessas escoradas com os rotulos maduros, score e decisao."""
    frames: list[pd.DataFrame] = []
    for delivery in manifest["deliveries"]:
        if not delivery.get("labels_file"):
            continue
        batch = read_table(settings.production_dir / delivery["batch_file"])
        labels = read_table(settings.production_dir / delivery["labels_file"])
        frame = add_engineered_features(batch.merge(labels, on=ID_COLUMN, validate="one_to_one"))
        frame[SCORE] = scorer.score(frame)
        frame[DECISION] = scorer.decide(frame[SCORE].to_numpy())
        frame["phase"] = delivery["phase"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _periods(reference_holdout: pd.DataFrame, production: pd.DataFrame) -> dict[str, pd.DataFrame]:
    periods = {REFERENCE_PERIOD: reference_holdout}
    for phase in production["phase"].drop_duplicates():
        periods[phase] = production[production["phase"] == phase]
    return periods


def _labeled_batches(pool: pd.DataFrame, plans: list[MonthPlan], n: int, seed: int) -> pd.DataFrame:
    """Lotes rotulados de ``plans`` gerados de ``pool``, com o cliente de origem de cada proposta."""
    frames = [
        generate_batch(pool, plan, n, seed).merge(
            batch_lineage(pool, plan, n, seed), on=ID_COLUMN, validate="one_to_one"
        )
        for plan in plans
    ]
    return pd.concat(frames, ignore_index=True)


def remediation_backtest(
    settings: Settings,
    scorer: CreditScorer,
    reference: pd.DataFrame,
    pool: pd.DataFrame,
    plans: list[MonthPlan],
    segment_cut: float,
    label_lag_months: int | None = None,
    with_ceiling: bool = False,
) -> dict:
    """Backtest das estrategias de remediacao, sem vazamento temporal nem de cliente.

    Os lotes de monitoramento sao sorteados todos do mesmo pool, entao o mesmo cliente aparece em
    varios meses. Para o backtest, o pool e dividido ao meio por cliente: os meses de treino sao
    regerados de uma metade e os de teste da outra, com metade do tamanho de lote (a mesma fracao
    de amostragem do monitoramento, logo a mesma intensidade de deslocamento). O experimento roda
    nas duas direcoes e as metricas sao a media das duas dobras.

    Os meses de teste sao sempre os ultimos ``settings.label_lag_months``. So entram no treino os
    meses cujo rotulo ja estaria maduro no primeiro mes de teste; ``label_lag_months`` permite
    simular um rotulo que amadurece mais rapido do que o configurado, mantendo o mesmo teste.
    ``with_ceiling`` acrescenta o teto: um modelo treinado nos proprios meses de teste.
    """
    window = max(settings.label_lag_months, 1)
    lag = window if label_lag_months is None else label_lag_months
    months = [plan.month for plan in plans]
    test_months = months[-window:]
    train_months = [month for month in months if add_months(month, lag) <= test_months[0]]
    train_plans = [plan for plan in plans if plan.month in train_months]
    test_plans = [plan for plan in plans if plan.month in test_months]

    order = np.random.default_rng(settings.seed).permutation(len(pool))
    halves = (pool.iloc[order[: len(pool) // 2]], pool.iloc[order[len(pool) // 2 :]])
    n = settings.batch_size // 2
    folds = []
    for train_pool, test_pool in (halves, halves[::-1]):
        folds.append(
            remediation_experiment(
                champion=scorer.model,
                champion_threshold=scorer.threshold,
                model_type=scorer.metadata.model_type,
                reference_train=reference[reference[SPLIT_COLUMN] == "train"],
                train_production=_labeled_batches(train_pool, train_plans, n, settings.seed),
                test_production=_labeled_batches(test_pool, test_plans, n, settings.seed),
                recent_months=train_months[-RECENT_WINDOW_MONTHS:],
                segment_cut=segment_cut,
                seed=settings.seed,
                group_column=SOURCE_ID,
                label_lag_months=lag,
                oracle_production=_labeled_batches(train_pool, test_plans, n, settings.seed) if with_ceiling else None,
            )
        )
    return {**average_folds(folds), "batch_size": n}


def _signed(value: float, std: float | None = None, digits: int = 3) -> str:
    text = f"{value:+.{digits}f}"
    return f"{text} ± {std:.{digits}f}" if std is not None else text


def _markdown(causal: dict, diagnosis: dict, fairness: dict, remediation: dict) -> str:
    """Resumo legivel das analises (as tabelas do README saem daqui)."""
    shapley, deviation, total = causal["shapley"], causal["shapley_std"], causal["total_effect"]
    lines = [
        "# Análises de governança: causalidade, viés e remediação",
        "",
        "Gerado por `python -m creditwatch.pipelines.analyze`. Os números mudam se o cenário ou a seed mudarem.",
        "",
        "## 1. O que causou a degradação (atribuição por mecanismo)",
        "",
        f"Intervenções sobre o plano de {causal['plan']['month']}: as {2 ** len(causal['mechanisms'])} combinações de "
        f"mecanismos, em lotes de {causal['sample_size']} propostas (o tamanho das remessas de produção), "
        f"repetidas com {len(causal['seeds'])} seeds. Cada célula é a parcela do efeito total atribuída ao mecanismo "
        "(valor de Shapley, média ± desvio-padrão entre as seeds).",
        "",
        "| Mecanismo | ROC AUC | Inadimplência entre aprovados | Taxa de aprovação | Features com drift |",
        "|---|---|---|---|---|",
    ]
    for name in causal["mechanisms"]:
        lines.append(
            f"| {MECHANISM_LABELS[name]} | {_signed(shapley['roc_auc'][name], deviation['roc_auc'][name])} | "
            f"{_signed(shapley['bad_rate_approved'][name], deviation['bad_rate_approved'][name])} | "
            f"{_signed(shapley['approval_rate'][name], deviation['approval_rate'][name])} | "
            f"{_signed(shapley['n_drifted'][name], deviation['n_drifted'][name], 1)} |"
        )
    lines += [
        f"| **Efeito total** | **{_signed(total['roc_auc'])}** | **{_signed(total['bad_rate_approved'])}** | "
        f"**{_signed(total['approval_rate'])}** | **{_signed(total['n_drifted'], digits=1)}** |",
        "",
        f"Sem nenhum mecanismo: ROC AUC {causal['baseline']['roc_auc']:.3f}, "
        f"{causal['baseline']['n_drifted']:.1f} features com drift. "
        f"Com todos: ROC AUC {causal['full']['roc_auc']:.3f}, {causal['full']['n_drifted']:.1f} features com drift.",
        "",
        "## 2. Onde procurar: varredura de calibração por feature",
        "",
        "Para cada faixa de cada feature (categorias ou quartis da referência), o excesso de erro é a diferença "
        "entre inadimplência observada e prevista no período de drift de conceito, menos a mesma diferença na "
        "referência. A dispersão é o maior excesso menos o menor entre as faixas da feature: quanto maior, mais a "
        "feature separa quem o modelo passou a errar de quem ele continua acertando.",
        "",
        "| Feature | Dispersão do excesso | Faixa com maior excesso | Excesso na faixa | Parcela da carteira |",
        "|---|---|---|---|---|",
    ]
    for row in diagnosis["calibration_scan"]["by_feature"][:8]:
        lines.append(
            f"| `{row['feature']}` | {row['spread']:.3f} | {row['worst_bin']} | {row['worst_excess_gap']:+.3f} | "
            f"{row['worst_bin_share']:.1%} |"
        )
    lines += [
        "",
        "## 3. Inadimplência observada x prevista por segmento",
        "",
        f"Segmentos montados com as variáveis que a varredura aponta: situação do pagamento mais recente e "
        f"utilização do limite, cortada na mediana da referência ({diagnosis['utilization_cut']:.2f}).",
        "",
        "| Período | Segmento | Participação | Observada | Prevista | Diferença |",
        "|---|---|---|---|---|---|",
    ]
    for row in diagnosis["segments"]:
        lines.append(
            f"| {PERIOD_LABELS.get(row['period'], row['period'])} | {row['segment']} | {row['share']:.1%} | "
            f"{row['observed_default_rate']:.1%} | {row['predicted_default_rate']:.1%} | {row['gap']:+.1%} |"
        )
    lines += [
        "",
        "## 4. Features candidatas a causa do drift de predição (PSI x importância)",
        "",
        "| Feature | PSI | Importância | Impacto |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| `{row['feature']}` | {row['psi']:.3f} | {row['importance']:.3f} | {row['impact']:.4f} |"
        for row in diagnosis["root_cause_ranking"]
    ]
    lines += [
        "",
        "## 5. Auditoria de viés (grupo menos aprovado de cada atributo)",
        "",
        "Impacto desigual: taxa de aprovação do grupo dividida pela do grupo mais aprovado, com intervalo de 95%. "
        "Abaixo de 0,80 indica impacto adverso (regra dos quatro quintos); se só o intervalo cruza 0,80, o resultado é "
        "inconclusivo. Diferença de oportunidade: quanto a taxa de aprovação entre bons pagadores do grupo fica "
        "abaixo da do grupo mais favorecido.",
        "",
        "| Período | Atributo | Grupo | n | Impacto desigual (IC 95%) | Diferença de oportunidade | Situação |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in fairness["worst_groups"]:
        low, high = row["disparate_impact_ci"]
        lines.append(
            f"| {PERIOD_LABELS.get(row['period'], row['period'])} | {ATTRIBUTE_LABELS[row['attribute']]} | "
            f"{row['group']} | {row['n']} | {row['disparate_impact']:.2f} ({low:.2f} a {high:.2f}) | "
            f"{row['equal_opportunity_gap'] * 100:+.1f} p.p. | {row['verdict']} |"
        )
    largest = fairness["largest_equal_opportunity_gap"]
    ablation = fairness["protected_attribute_ablation"]
    lines += [
        "",
        "Maior diferença de oportunidade entre todos os grupos comparados: "
        f"{largest['equal_opportunity_gap'] * 100:.1f} p.p. ({ATTRIBUTE_LABELS[largest['attribute']].lower()}: "
        f"{largest['group']}, {PERIOD_LABELS.get(largest['period'], largest['period']).lower()}).",
        "",
        "### Quanto o modelo ganharia usando os atributos protegidos",
        "",
        f"Mesmo algoritmo ({ablation['model_type']}), {len(ablation['seeds'])} seeds, com e sem "
        f"{', '.join(ablation['protected_attributes'])} como preditores.",
        "",
        "| Variante | ROC AUC (validação) | ROC AUC (teste) |",
        "|---|---|---|",
        f"| Sem atributos protegidos (modelo em produção) | {ablation['without_protected']['roc_auc_val']:.4f} | "
        f"{ablation['without_protected']['roc_auc_test']:.4f} |",
        f"| Com atributos protegidos | {ablation['with_protected']['roc_auc_val']:.4f} | "
        f"{ablation['with_protected']['roc_auc_test']:.4f} |",
        f"| **Ganho** | **{ablation['gain_val']:+.4f}** | **{ablation['gain_test']:+.4f}** |",
        "",
        "## 6. Remediação (backtest em meses posteriores ao ajuste)",
        "",
        f"Ajuste com os rótulos maduros de {remediation['train_months'][0]} a {remediation['train_months'][-1]} "
        f"(atraso de {remediation['label_lag_months']} meses); teste em {', '.join(remediation['test_months'])}. "
        f"O pool de clientes é dividido ao meio: uma metade gera os meses de ajuste e a outra os de teste "
        f"({remediation['batch_size']} propostas por mês), e o experimento roda nas duas direções "
        f"({remediation['test_rows']} propostas de teste no total). Nenhum cliente aparece nos dois lados.",
        "",
        "| Estratégia | Linhas de ajuste | ROC AUC (dobras) | KS | Aprovação no corte próprio | "
        "Inadimplência entre aprovados | Inadimplência entre aprovados, mesmo volume |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, row in remediation["strategies"].items():
        by_fold = " / ".join(f"{value:.3f}" for value in row["roc_auc_by_fold"])
        lines.append(
            f"| {STRATEGY_LABELS[name]} | {row['train_rows']} | {row['roc_auc']:.3f} ({by_fold}) | {row['ks']:.3f} | "
            f"{row['approval_rate']:.1%} | {row['bad_rate_approved']:.1%} | "
            f"{row['bad_rate_at_champion_approval']:.1%} |"
        )
    factors = ", ".join(
        f"{segment} {factor:.2f}" for segment, factor in remediation["segment_calibration_factors"].items()
    )
    lines += [
        "",
        f"A última coluna aprova o mesmo volume em todas as estratégias ({remediation['champion_approval_rate']:.1%}, "
        "a taxa de aprovação do modelo atual nos meses de teste), o que torna a inadimplência comparável.",
        "",
        f"Fatores da recalibração por segmento (inadimplência observada ÷ prevista nas safras recentes): {factors}.",
        "",
        "O teto é um modelo treinado nos próprios meses de teste, com outros clientes: mostra quanto um retreino "
        "recuperaria se o rótulo fosse imediato. A distância entre o teto e o desempenho da referência é a parte "
        "da perda que nenhum retreino recupera, porque o drift de conceito tornou a inadimplência menos previsível.",
    ]
    shorter = remediation.get("shorter_label_lag")
    if shorter:
        lines += [
            "",
            f"### Se o rótulo amadurecesse em {shorter['label_lag_months']} mês",
            "",
            f"Mesmo teste, com ajuste até {shorter['train_months'][-1]}.",
            "",
            "| Estratégia | ROC AUC (dobras) | Inadimplência entre aprovados, mesmo volume |",
            "|---|---|---|",
        ]
        for name, row in shorter["strategies"].items():
            by_fold = " / ".join(f"{value:.3f}" for value in row["roc_auc_by_fold"])
            lines.append(
                f"| {STRATEGY_LABELS[name]} | {row['roc_auc']:.3f} ({by_fold}) | "
                f"{row['bad_rate_at_champion_approval']:.1%} |"
            )
    return "\n".join(lines) + "\n"


def run_analyze(
    settings: Settings, tracker: StageTracker | None = None, attribution_seeds: int = ATTRIBUTION_SEEDS
) -> dict:
    tracker = tracker or StageTracker("analyze", settings.events_path)
    analysis_dir = settings.reports_dir / "analysis"

    with tracker.stage("load_data") as event:
        manifest = read_json(settings.production_dir / MANIFEST_FILENAME)
        scorer = CreditScorer.load(settings.models_dir)
        reference = add_engineered_features(read_table(settings.reference_path))
        holdout = reference[reference[SPLIT_COLUMN] != "train"].copy()
        holdout[SCORE] = scorer.score(holdout)
        holdout[DECISION] = scorer.decide(holdout[SCORE].to_numpy())
        pool = read_table(settings.production_pool_path)
        production = load_labeled_production(settings, scorer, manifest)
        event.rows_out = len(production)

    periods = _periods(holdout, production)
    scored_deliveries = [delivery for delivery in manifest["deliveries"] if delivery.get("labels_file")]
    plans = [MonthPlan(**delivery["plan"]) for delivery in scored_deliveries]
    cut = utilization_cut(holdout)

    with tracker.stage("causal_attribution") as event:
        thresholds = DriftThresholds(
            settings.psi_warning,
            settings.psi_critical,
            settings.ks_effect_size,
            settings.p_value_alpha,
            settings.drift_share_alert,
        )
        seeds = tuple(settings.seed + offset for offset in range(attribution_seeds))
        causal = mechanism_attribution(
            pool, plans[-1], scorer, reference, holdout, manifest["batch_size"], seeds, thresholds
        )
        dominant = min(causal["shapley"]["roc_auc"], key=causal["shapley"]["roc_auc"].get)
        event.details = {
            "dominant_mechanism_auc": dominant,
            "combinations": 2 ** len(causal["mechanisms"]),
            "seeds": len(seeds),
        }

    with tracker.stage("calibration_diagnosis") as event:
        last_phase = production["phase"].iloc[-1]
        last_delivery = scored_deliveries[-1]["delivery"]
        last_batch = read_json(settings.reports_dir / "monitoring" / "batches" / f"{last_delivery}.json")
        diagnosis = {
            "utilization_cut": round(cut, 4),
            "scan_period": last_phase,
            "calibration_scan": calibration_scan(holdout, periods[last_phase], MODEL_FEATURES),
            "segments": segment_diagnosis(periods, cut),
            "root_cause_ranking": root_cause_ranking(
                last_batch["drift"]["features"], scorer.metadata.feature_importance
            ),
        }
        event.details = {"top_feature": diagnosis["calibration_scan"]["by_feature"][0]["feature"]}

    with tracker.stage("fairness_audit") as event:
        fairness = fairness_audit(periods)
        fairness["protected_attribute_ablation"] = protected_attribute_ablation(
            reference, scorer.metadata.model_type, seeds=(settings.seed, 7, 123)
        )
        event.details = {
            "adverse_impact_groups": sum(row["adverse_impact"] for row in fairness["worst_groups"]),
            "protected_attributes_gain_test": fairness["protected_attribute_ablation"]["gain_test"],
        }

    with tracker.stage("remediation_backtest") as event:
        remediation = remediation_backtest(settings, scorer, reference, pool, plans, cut, with_ceiling=True)
        # Sensibilidade: o mesmo teste, se o rotulo amadurecesse um mes mais cedo.
        shorter_lag = remediation["label_lag_months"] - 1
        remediation["shorter_label_lag"] = (
            remediation_backtest(settings, scorer, reference, pool, plans, cut, label_lag_months=shorter_lag)
            if shorter_lag >= 1
            else None
        )
        event.details = {name: row["roc_auc"] for name, row in remediation["strategies"].items()}

    write_json(analysis_dir / "causal_attribution.json", {**causal, "mechanism_labels": MECHANISM_LABELS})
    write_json(analysis_dir / "segment_diagnosis.json", diagnosis)
    write_json(analysis_dir / "fairness_audit.json", fairness)
    write_json(analysis_dir / "remediation.json", {**remediation, "strategy_labels": STRATEGY_LABELS})
    (analysis_dir / "analysis_report.md").write_text(
        _markdown(causal, diagnosis, fairness, remediation), encoding="utf-8", newline="\n"
    )
    logger.info("analises gravadas em %s (mecanismo dominante na queda de AUC: %s)", analysis_dir, dominant)
    return {
        "causal": causal,
        "diagnosis": diagnosis,
        "segments": diagnosis["segments"],
        "fairness": fairness,
        "remediation": remediation,
    }


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    parser = build_parser("Analise causal, auditoria de vies e backtest de remediacao.")
    parser.add_argument("--attribution-seeds", type=int, default=ATTRIBUTION_SEEDS)
    args = parser.parse_args(argv)
    start_run(settings, args.log_level)
    run_analyze(settings, attribution_seeds=args.attribution_seeds)


if __name__ == "__main__":
    main()
