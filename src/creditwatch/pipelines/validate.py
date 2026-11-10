"""Validacao avulsa de um arquivo contra o contrato de dados (o "portao" em linha de comando).

Uso tipico: checar um lote antes de envia-lo ao pipeline ou demonstrar o bloqueio de um lote ruim.
O codigo de saida e 0 se o lote passa e 1 se e bloqueado, o que permite usar o comando como
portao em um orquestrador ou em CI::

    python -m creditwatch.pipelines.validate data/production/batch_2025-07_remessa1_corrompida.parquet
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from creditwatch.config.settings import Settings, get_settings
from creditwatch.contracts.credit_contract import build_contract
from creditwatch.contracts.gate import ContractResult, quarantine_batch, validate_batch, write_validation_report
from creditwatch.pipelines._cli import build_parser, read_table, start_run

logger = logging.getLogger("creditwatch.pipelines.validate")


def format_result(result: ContractResult) -> str:
    """Tabela em texto com o veredito e as regras violadas."""
    lines = [
        f"Contrato: {result.contract} | Lote: {result.batch} | {result.n_rows} linhas x {result.n_columns} colunas",
        f"Resultado: {'APROVADO' if result.passed else 'BLOQUEADO'}",
    ]
    if result.violations:
        lines.append(f"Linhas com pelo menos uma falha: {result.rows_with_failures}")
        lines.append("")
        lines.append(f"{'Regra':<7} {'Falhas':>7}  {'Titulo':<26} Colunas")
        for violation in result.violations:
            columns = ", ".join(violation.columns) or "-"
            lines.append(f"{violation.rule_id:<7} {violation.n_failures:>7}  {violation.title:<26} {columns}")
    for warning in result.warnings:
        lines.append(f"Aviso: {warning}")
    return "\n".join(lines)


def run_validate(
    path: Path, settings: Settings, require_target: bool = False, min_rows: int = 500, quarantine: bool = True
) -> ContractResult:
    """Valida o arquivo, grava o relatorio e, se reprovado, move o lote para a quarentena."""
    path = Path(path)
    frame = read_table(path)
    contract = build_contract(require_target, min_rows)
    result = validate_batch(frame, contract, batch=path.stem)
    report_path = write_validation_report(
        result, settings.reports_dir / "data_quality" / f"validation_{path.stem}.json"
    )
    if not result.passed and quarantine:
        quarantine_batch(frame, result, settings.quarantine_dir, set(contract.columns))
    logger.info("relatorio de validacao gravado em %s", report_path)
    return result


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    parser = build_parser("Valida um arquivo (Parquet ou CSV) contra o contrato de dados.")
    parser.add_argument("path", type=Path, help="Arquivo do lote a validar.")
    parser.add_argument("--require-target", action="store_true", help="Exige a coluna de rotulo (referencia).")
    parser.add_argument("--min-rows", type=int, default=500)
    parser.add_argument("--no-quarantine", action="store_true", help="Nao copia o lote reprovado para a quarentena.")
    args = parser.parse_args(argv)
    start_run(settings, args.log_level)
    result = run_validate(args.path, settings, args.require_target, args.min_rows, quarantine=not args.no_quarantine)
    print(format_result(result))
    if not result.passed:
        logger.error("ingestao BLOQUEADA: %s", ", ".join(result.violated_rules))
        sys.exit(1)


if __name__ == "__main__":
    main()
