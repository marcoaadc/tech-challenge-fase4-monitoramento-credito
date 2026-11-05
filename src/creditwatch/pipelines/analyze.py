"""Estagio 5 - analises: causalidade, vies e remediacao.

Usa os lotes de producao ja validados e os seus rotulos maduros para responder a tres perguntas
de governanca: *o que causou a degradacao*, *quem foi afetado de forma desigual* e *o que a
corrige*. Os resultados vao para ``reports/analysis/`` (JSON e um resumo em Markdown).
"""

from __future__ import annotations

import logging

import pandas as pd

from creditwatch.analysis.causal import mechanism_attribution, root_cause_ranking, segment_diagnosis
from creditwatch.analysis.fairness import fairness_audit, protected_attribute_ablation
from creditwatch.analysis.remediation import remediation_experiment
from creditwatch.config.settings import Settings, get_settings
from creditwatch.data.schema import BATCH_MONTH, DECISION, ID_COLUMN, SCORE
from creditwatch.data.split import SPLIT_COLUMN
from creditwatch.drift.detector import DriftThresholds
from creditwatch.features.engineering import add_engineered_features
from creditwatch.model.registry import CreditScorer
from creditwatch.observability.events import StageTracker
from creditwatch.pipelines._cli import build_parser, read_json, read_table, start_run, write_json
from creditwatch.pipelines.simulate import MANIFEST_FILENAME
from creditwatch.simulation.scenario import MonthPlan

logger = logging.getLogger("creditwatch.pipelines.analyze")

REFERENCE_PERIOD = "referencia"
PERIOD_LABELS = {
    "referencia": "Referência",
    "estavel": "Estável",
    "drift_de_dados": "Drift de dados",
    "drift_de_conceito": "Drift de conceito",
}
ATTRIBUTE_LABELS = {"sex": "Sexo", "age_band": "Faixa etária", "education": "Escolaridade", "marriage": "Estado civil"}
RECENT_WINDOW_MONTHS = 3
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


def _markdown(causal: dict, segments: list[dict], ranking: list[dict], fairness: dict, remediation: dict) -> str:
    """Resumo legivel das analises (as tabelas do README saem daqui)."""
    lines = ["# Análises de governança: causalidade, viés e remediação", ""]
    lines += [
        "Gerado por `python -m creditwatch.pipelines.analyze`. Os números mudam se o cenário ou a seed mudarem.",
        "",
        "## 1. O que causou a degradação (atribuição por mecanismo)",
        "",
        f"Intervenções sobre o plano de {causal['plan']['month']} ({causal['sample_size']} propostas por combinação). "
        "Cada célula é a parcela do efeito total atribuída ao mecanismo (valores de Shapley).",
        "",
        "| Mecanismo | ROC AUC | KS | Inadimplência | Inadimplência entre aprovados | Features com drift |",
        "|---|---|---|---|---|---|",
    ]
    shapley = causal["shapley"]
    for name in causal["mechanisms"]:
        lines.append(
            f"| {MECHANISM_LABELS[name]} | {shapley['roc_auc'][name]:+.3f} | {shapley['ks'][name]:+.3f} | "
            f"{shapley['default_rate'][name]:+.3f} | {shapley['bad_rate_approved'][name]:+.3f} | "
            f"{shapley['n_drifted'][name]:+.1f} |"
        )
    total = causal["total_effect"]
    lines += [
        f"| **Efeito total** | **{total['roc_auc']:+.3f}** | **{total['ks']:+.3f}** | "
        f"**{total['default_rate']:+.3f}** | **{total['bad_rate_approved']:+.3f}** | **{total['n_drifted']:+.1f}** |",
        "",
        "## 2. Onde a relação mudou (inadimplência observada x prevista por segmento)",
        "",
        "| Período | Segmento | Participação | Observada | Prevista | Diferença |",
        "|---|---|---|---|---|---|",
    ]
    for row in segments:
        lines.append(
            f"| {PERIOD_LABELS.get(row['period'], row['period'])} | {row['segment']} | {row['share']:.1%} | "
            f"{row['observed_default_rate']:.1%} | "
            f"{row['predicted_default_rate']:.1%} | {row['gap']:+.1%} |"
        )
    lines += [
        "",
        "## 3. Features candidatas a causa do drift de predição (PSI x importância)",
        "",
        "| Feature | PSI | Importância | Impacto |",
        "|---|---|---|---|",
    ]
    lines += [f"| `{r['feature']}` | {r['psi']:.3f} | {r['importance']:.3f} | {r['impact']:.4f} |" for r in ranking]
    lines += [
        "",
        "## 4. Auditoria de viés (grupo menos aprovado de cada atributo)",
        "",
        "| Período | Atributo | Grupo | Impacto desigual | Diferença de oportunidade | Impacto adverso (< 0,80) |",
        "|---|---|---|---|---|---|",
    ]
    for row in fairness["worst_groups"]:
        lines.append(
            f"| {PERIOD_LABELS.get(row['period'], row['period'])} | {ATTRIBUTE_LABELS[row['attribute']]} | "
            f"{row['group']} | {row['disparate_impact']:.2f} | "
            f"{row['equal_opportunity_gap']:+.3f} | {'sim' if row['adverse_impact'] else 'não'} |"
        )
    ablation = fairness["protected_attribute_ablation"]
    lines += [
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
        "## 5. Remediação (avaliada em meses posteriores ao treino)",
        "",
        f"Treino com rótulos maduros de {remediation['train_months'][0]} a {remediation['train_months'][-1]}; "
        f"teste em {', '.join(remediation['test_months'])} ({remediation['test_rows']} propostas).",
        "",
        "| Estratégia | Linhas de treino | ROC AUC | KS | Aprovação | Inadimplência entre aprovados |",
        "|---|---|---|---|---|---|",
    ]
    for name, row in remediation["strategies"].items():
        lines.append(
            f"| `{name}` | {row['train_rows']} | {row['roc_auc']:.3f} | {row['ks']:.3f} | {row['approval_rate']:.1%} | "
            f"{row['bad_rate_approved']:.1%} |"
        )
    return "\n".join(lines) + "\n"


def run_analyze(settings: Settings, tracker: StageTracker | None = None, attribution_sample: int = 6000) -> dict:
    tracker = tracker or StageTracker("analyze", settings.events_path)
    analysis_dir = settings.reports_dir / "analysis"

    with tracker.stage("load_data") as event:
        manifest = read_json(settings.production_dir / MANIFEST_FILENAME)
        scorer = CreditScorer.load(settings.models_dir)
        reference = add_engineered_features(read_table(settings.reference_path))
        holdout = reference[reference[SPLIT_COLUMN] != "train"].copy()
        holdout[SCORE] = scorer.score(holdout)
        holdout[DECISION] = scorer.decide(holdout[SCORE].to_numpy())
        production = load_labeled_production(settings, scorer, manifest)
        pool = read_table(settings.production_pool_path)
        event.rows_out = len(production)

    periods = _periods(holdout, production)
    months = sorted(production[BATCH_MONTH].unique())

    with tracker.stage("causal_attribution") as event:
        final_plan = MonthPlan(**manifest["deliveries"][-1]["plan"])
        thresholds = DriftThresholds(
            settings.psi_warning,
            settings.psi_critical,
            settings.ks_effect_size,
            settings.p_value_alpha,
            settings.drift_share_alert,
        )
        causal = mechanism_attribution(
            pool, final_plan, scorer, reference, holdout, attribution_sample, settings.seed, thresholds
        )
        segments = segment_diagnosis(periods)
        last_batch = read_json(settings.reports_dir / "monitoring" / "batches" / f"{months[-1]}.json")
        ranking = root_cause_ranking(last_batch["drift"]["features"], scorer.metadata.feature_importance)
        dominant = min(causal["shapley"]["roc_auc"], key=causal["shapley"]["roc_auc"].get)
        event.details = {"dominant_mechanism_auc": dominant, "combinations": 2 ** len(causal["mechanisms"])}

    with tracker.stage("fairness_audit") as event:
        fairness = fairness_audit(periods)
        fairness["protected_attribute_ablation"] = protected_attribute_ablation(
            reference, scorer.metadata.model_type, seeds=(settings.seed, 7, 123)
        )
        event.details = {
            "adverse_impact_groups": sum(row["adverse_impact"] for row in fairness["worst_groups"]),
            "protected_attributes_gain_test": fairness["protected_attribute_ablation"]["gain_test"],
        }

    with tracker.stage("remediation_experiment") as event:
        lag = max(settings.label_lag_months, 1)
        train_months, test_months = months[:-lag], months[-lag:]
        remediation = remediation_experiment(
            champion=scorer.model,
            champion_threshold=scorer.threshold,
            model_type=scorer.metadata.model_type,
            reference_train=reference[reference[SPLIT_COLUMN] == "train"],
            production=production,
            train_months=train_months,
            recent_months=train_months[-RECENT_WINDOW_MONTHS:],
            test_months=test_months,
            seed=settings.seed,
        )
        event.details = {name: row["roc_auc"] for name, row in remediation["strategies"].items()}

    write_json(analysis_dir / "causal_attribution.json", {**causal, "mechanism_labels": MECHANISM_LABELS})
    write_json(analysis_dir / "segment_diagnosis.json", {"segments": segments, "root_cause_ranking": ranking})
    write_json(analysis_dir / "fairness_audit.json", fairness)
    write_json(analysis_dir / "remediation.json", remediation)
    (analysis_dir / "analysis_report.md").write_text(
        _markdown(causal, segments, ranking, fairness, remediation), encoding="utf-8", newline="\n"
    )
    logger.info("analises gravadas em %s (mecanismo dominante na queda de AUC: %s)", analysis_dir, dominant)
    return {"causal": causal, "segments": segments, "fairness": fairness, "remediation": remediation}


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    parser = build_parser("Analise causal, auditoria de vies e experimento de remediacao.")
    parser.add_argument("--attribution-sample", type=int, default=6000)
    args = parser.parse_args(argv)
    start_run(settings, args.log_level)
    run_analyze(settings, attribution_sample=args.attribution_sample)


if __name__ == "__main__":
    main()
