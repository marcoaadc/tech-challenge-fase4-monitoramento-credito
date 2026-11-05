"""Catalogo das regras de qualidade do contrato de dados.

Cada regra tem um identificador estavel (``DQ-xx``) que aparece nos relatorios de validacao, nos
logs, nas metricas Prometheus e na documentacao, permitindo rastrear uma falha ate a sua regra.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

BLOCKING, WARNING = "bloqueante", "aviso"


@dataclass(frozen=True)
class Rule:
    rule_id: str
    title: str
    description: str
    severity: str = BLOCKING

    def as_dict(self) -> dict:
        return asdict(self)


RULES: tuple[Rule, ...] = (
    Rule("DQ-01", "Completude", "Nenhuma coluna obrigatória pode conter valores nulos."),
    Rule("DQ-02", "Idade válida", "Idade entre 18 e 100 anos (capacidade civil para contratar crédito)."),
    Rule("DQ-03", "Limite de crédito", "Limite entre 1.000 e 1.000.000 (teto da política de crédito)."),
    Rule("DQ-04", "Unicidade", "customer_id não pode se repetir dentro do lote."),
    Rule("DQ-05", "Domínio dos categóricos", "sex, education e marriage restritos ao dicionário de dados."),
    Rule("DQ-06", "Status de pagamento", "pay_status_1..6 inteiros entre -2 e 8."),
    Rule("DQ-07", "Pagamentos não negativos", "pay_amt_1..6 maiores ou iguais a zero."),
    Rule("DQ-08", "Faturas plausíveis", "bill_amt_1..6 entre -1.000.000 e 10.000.000 (erro de unidade/escala)."),
    Rule("DQ-09", "Esquema estrito", "Todas as colunas do contrato presentes e nenhuma coluna extra (ex.: PII)."),
    Rule("DQ-10", "Tipos", "Colunas numéricas devem ser inteiras (sem texto ou fração)."),
    Rule("DQ-11", "Volume mínimo", "O lote deve ter ao menos o número mínimo de linhas configurado."),
    Rule("DQ-12", "Rótulo binário", "Quando presente, default só admite 0 ou 1."),
    Rule("DQ-13", "Mês de referência", "reference_month no formato AAAA-MM e único dentro do lote."),
    Rule(
        "DQ-W01",
        "Carteira inativa",
        "Menos de 60% dos clientes sem consumo no mês mais recente (pay_status_1 = -2).",
        severity=WARNING,
    ),
)
RULES_BY_ID = {rule.rule_id: rule for rule in RULES}

# Falhas estruturais do Pandera -> regra do catalogo.
BUILTIN_CHECK_RULES = {
    "not_nullable": "DQ-01",
    "field_uniqueness": "DQ-04",
    "multiple_fields_uniqueness": "DQ-04",
    "column_in_schema": "DQ-09",
    "column_in_dataframe": "DQ-09",
    "coerce_dtype": "DQ-10",
    "dtype": "DQ-10",
}


def rule_id_for_check(check: str) -> str:
    """Resolve o identificador da regra a partir do texto do check reportado pelo Pandera."""
    check = str(check)
    if check.startswith("DQ-"):
        return check.split(":", 1)[0].strip()
    for prefix, rule_id in BUILTIN_CHECK_RULES.items():
        if check.startswith(prefix):
            return rule_id
    return "DQ-??"
