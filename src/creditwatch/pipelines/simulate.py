"""Estagio 3 - simulacao da producao: doze remessas mensais com degradacao progressiva.

Cada mes gera dois arquivos, como aconteceria em producao:

- ``batch_<mes>.parquet``: as propostas a escorar, **sem rotulo**;
- ``labels_<mes>.parquet``: a inadimplencia observada, que so chega apos a maturacao.

Uma das remessas e entregue corrompida pelo sistema de origem (e reenviada corrigida em seguida),
para que o bloqueio do contrato apareca na linha do tempo do monitoramento.
"""

from __future__ import annotations

import logging

from creditwatch.config.settings import Settings, get_settings
from creditwatch.data.schema import ID_COLUMN, TARGET
from creditwatch.observability.events import StageTracker
from creditwatch.pipelines._cli import build_parser, read_table, start_run, write_json, write_table
from creditwatch.simulation.corruption import DEMO_DEFECTS, corrupt_batch, expected_rules
from creditwatch.simulation.scenario import MonthPlan, default_scenario, generate_batch

logger = logging.getLogger("creditwatch.pipelines.simulate")

MANIFEST_FILENAME = "manifest.json"
CORRUPTED_MONTH = "2025-07"  # a primeira remessa deste mes chega corrompida


def run_simulate(
    settings: Settings,
    tracker: StageTracker | None = None,
    scenario: list[MonthPlan] | None = None,
    corrupted_month: str | None = CORRUPTED_MONTH,
) -> dict:
    tracker = tracker or StageTracker("simulate", settings.events_path)
    scenario = scenario or default_scenario()
    production_dir = settings.production_dir
    pool = read_table(settings.production_pool_path)

    deliveries: list[dict] = []
    for plan in scenario:
        with tracker.stage("generate_batch", batch=plan.month) as event:
            batch = generate_batch(pool, plan, settings.batch_size, settings.seed)
            features = batch.drop(columns=[TARGET])
            labels = batch[[ID_COLUMN, TARGET]]

            if plan.month == corrupted_month:
                corrupted = corrupt_batch(features, DEMO_DEFECTS, seed=settings.seed)
                corrupted_file = write_table(
                    corrupted, production_dir / f"batch_{plan.month}_remessa1_corrompida.parquet"
                )
                deliveries.append(
                    {
                        "delivery": f"{plan.month}-r1",
                        "month": plan.month,
                        "phase": plan.phase,
                        "batch_file": corrupted_file.name,
                        "labels_file": None,
                        "rows": len(corrupted),
                        "note": "remessa corrompida pelo sistema de origem",
                        "injected_defects": list(DEMO_DEFECTS),
                        "expected_rules": expected_rules(DEMO_DEFECTS),
                        "plan": plan.as_dict(),
                    }
                )
                delivery_id, note = f"{plan.month}-r2", "reenvio corrigido"
            else:
                delivery_id, note = plan.month, ""

            batch_file = write_table(features, production_dir / f"batch_{plan.month}.parquet")
            labels_file = write_table(labels, production_dir / f"labels_{plan.month}.parquet")
            deliveries.append(
                {
                    "delivery": delivery_id,
                    "month": plan.month,
                    "phase": plan.phase,
                    "batch_file": batch_file.name,
                    "labels_file": labels_file.name,
                    "rows": len(features),
                    "note": note,
                    "plan": plan.as_dict(),
                }
            )
            event.rows_out = len(features)
            event.details = plan.as_dict()

    manifest = {
        "seed": settings.seed,
        "batch_size": settings.batch_size,
        "label_lag_months": settings.label_lag_months,
        "deliveries": deliveries,
    }
    write_json(production_dir / MANIFEST_FILENAME, manifest)
    logger.info("%d remessas geradas em %s", len(deliveries), production_dir)
    return manifest


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    parser = build_parser("Gera as remessas mensais de producao com drift de dados e de conceito.")
    args = parser.parse_args(argv)
    start_run(settings, args.log_level)
    run_simulate(settings)


if __name__ == "__main__":
    main()
