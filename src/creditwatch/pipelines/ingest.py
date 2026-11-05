"""Estagio 1 - ingestao: do arquivo bruto ao Dataset de Referencia validado.

1. Le o snapshot bruto (baixa da UCI se nao existir) e confere a integridade.
2. Valida o bruto contra o contrato de dados e registra o resultado: ele *reprova* (codigos fora
   do dicionario), o que demonstra o contrato barrando dado real sujo.
3. Aplica as regras de limpeza documentadas.
4. Exige que o dataset limpo passe no contrato (portao bloqueante).
5. Separa Referencia (treino) e populacao de Producao (nunca vista no treino).
"""

from __future__ import annotations

import logging

from creditwatch.config.settings import Settings, get_settings
from creditwatch.contracts.credit_contract import build_contract
from creditwatch.contracts.gate import enforce_contract, validate_batch, write_validation_report
from creditwatch.data.cleaning import clean_raw_dataset
from creditwatch.data.schema import TARGET
from creditwatch.data.source import UCI_DOI, UCI_URL, load_raw_dataset, sha256_of
from creditwatch.data.split import SPLIT_COLUMN, assign_model_splits, split_reference_production
from creditwatch.observability.events import StageTracker
from creditwatch.pipelines._cli import build_parser, start_run, write_json, write_table

logger = logging.getLogger("creditwatch.pipelines.ingest")


def run_ingest(settings: Settings, tracker: StageTracker | None = None) -> dict:
    tracker = tracker or StageTracker("ingest", settings.events_path)
    quality_dir = settings.reports_dir / "data_quality"
    contract = build_contract(require_target=True)

    with tracker.stage("load_raw") as event:
        raw = load_raw_dataset(settings.raw_dataset_path)
        raw_sha = sha256_of(settings.raw_dataset_path)
        event.rows_out = len(raw)

    with tracker.stage("validate_raw", rows_in=len(raw)) as event:
        raw_result = validate_batch(raw, contract, batch="raw")
        write_validation_report(raw_result, quality_dir / "raw_validation.json")
        event.details = {"status": raw_result.as_dict()["status"], "violated_rules": raw_result.violated_rules}
        logger.info(
            "dataset bruto %s no contrato (%d linhas com falha; regras: %s)",
            "aprovado" if raw_result.passed else "REPROVADO",
            raw_result.rows_with_failures,
            ", ".join(raw_result.violated_rules) or "-",
        )

    with tracker.stage("clean", rows_in=len(raw)) as event:
        clean, cleaning = clean_raw_dataset(raw)
        event.rows_out = len(clean)
        event.details = cleaning.as_dict()

    with tracker.stage("contract_gate", rows_in=len(clean)) as event:
        clean, clean_result = enforce_contract(
            clean, contract, batch="reference_candidate", quarantine_dir=settings.quarantine_dir
        )
        write_validation_report(clean_result, quality_dir / "clean_validation.json")
        event.rows_out = len(clean)

    with tracker.stage("split", rows_in=len(clean)) as event:
        reference, pool = split_reference_production(clean, settings.reference_fraction, settings.seed)
        reference = assign_model_splits(reference, seed=settings.seed)
        write_table(reference, settings.reference_path)
        write_table(pool, settings.production_pool_path)
        event.rows_out = len(reference) + len(pool)

    report = {
        "source": {
            "name": "Default of Credit Card Clients (UCI Machine Learning Repository)",
            "url": UCI_URL,
            "doi": UCI_DOI,
            "license": "CC BY 4.0",
            "raw_file": settings.raw_dataset_path.as_posix(),
            "raw_sha256": raw_sha,
        },
        "raw_validation": {
            "status": raw_result.as_dict()["status"],
            "violated_rules": raw_result.violated_rules,
            "rows_with_failures": raw_result.rows_with_failures,
        },
        "cleaning": cleaning.as_dict(),
        "clean_validation": {"status": clean_result.as_dict()["status"], "warnings": clean_result.warnings},
        "reference": {
            "path": settings.reference_path.as_posix(),
            "sha256": sha256_of(settings.reference_path),
            "rows": len(reference),
            "splits": {name: int(count) for name, count in reference[SPLIT_COLUMN].value_counts().items()},
            "default_rate": round(float(reference[TARGET].mean()), 4),
        },
        "production_pool": {
            "path": settings.production_pool_path.as_posix(),
            "rows": len(pool),
            "default_rate": round(float(pool[TARGET].mean()), 4),
        },
    }
    write_json(quality_dir / "ingestion_report.json", report)
    logger.info("referencia: %d linhas | pool de producao: %d linhas", len(reference), len(pool))
    return report


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    parser = build_parser("Ingestao: valida, limpa e separa referencia x producao.")
    args = parser.parse_args(argv)
    start_run(settings, args.log_level)
    run_ingest(settings)


if __name__ == "__main__":
    main()
