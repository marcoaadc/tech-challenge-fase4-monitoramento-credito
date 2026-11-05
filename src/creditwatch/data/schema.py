"""Fonte unica da verdade sobre as colunas do dataset.

Dataset: *Default of Credit Card Clients* (UCI, Yeh & Lien, 2009), 30.000 clientes de cartao de
credito em Taiwan (abr-set/2005). Os nomes originais sao normalizados para ``snake_case`` e o
historico mensal passa a ser indexado de 1 (mes mais recente) a 6 (mes mais antigo), corrigindo a
inconsistencia do arquivo original (``PAY_0, PAY_2, ..., PAY_6``).
"""

from __future__ import annotations

MONTHS = (1, 2, 3, 4, 5, 6)

ID_COLUMN = "customer_id"
TARGET = "default"
SCORE = "score"
DECISION = "approved"
BATCH_MONTH = "reference_month"

LIMIT = "limit_bal"
PAY_STATUS = tuple(f"pay_status_{m}" for m in MONTHS)
BILL_AMT = tuple(f"bill_amt_{m}" for m in MONTHS)
PAY_AMT = tuple(f"pay_amt_{m}" for m in MONTHS)

# Atributos sociodemograficos: NAO entram no modelo (privacy by design). Sao mantidos apenas
# para validar o contrato (ex.: maioridade) e para a auditoria de vies.
PROTECTED = ("sex", "education", "marriage", "age")

# Colunas que o contrato de dados exige em qualquer lote (referencia ou producao).
CONTRACT_COLUMNS = (ID_COLUMN, LIMIT, *PROTECTED, *PAY_STATUS, *BILL_AMT, *PAY_AMT)

# Features calculadas em ``creditwatch.features.engineering``.
ENGINEERED = (
    "utilization_1",
    "utilization_avg",
    "payment_ratio_1",
    "payment_ratio_avg",
    "max_delay",
    "months_delayed",
    "bill_trend",
)

# Entradas do modelo: apenas comportamento financeiro.
MODEL_FEATURES = (LIMIT, *PAY_STATUS, *BILL_AMT, *PAY_AMT, *ENGINEERED)

# Tipos para os testes estatisticos de drift.
CATEGORICAL_MONITORED = ("sex", "education", "marriage", *PAY_STATUS, "max_delay", "months_delayed")
NUMERICAL_MONITORED = (
    LIMIT,
    "age",
    *BILL_AMT,
    *PAY_AMT,
    "utilization_1",
    "utilization_avg",
    "payment_ratio_1",
    "payment_ratio_avg",
    "bill_trend",
)
MONITORED = (*NUMERICAL_MONITORED, *CATEGORICAL_MONITORED)

# Dominios documentados no dicionario de dados da UCI.
SEX_DOMAIN = {1: "masculino", 2: "feminino"}
EDUCATION_DOMAIN = {1: "pós-graduação", 2: "graduação", 3: "ensino médio", 4: "outros"}
MARRIAGE_DOMAIN = {1: "casado(a)", 2: "solteiro(a)", 3: "outros"}
PAY_STATUS_MIN, PAY_STATUS_MAX = -2, 8  # -2 sem consumo, -1 pago integralmente, 0 rotativo, 1..8 meses de atraso

# Mapeamento arquivo original -> nomes normalizados.
RAW_RENAME = {
    "ID": ID_COLUMN,
    "LIMIT_BAL": LIMIT,
    "SEX": "sex",
    "EDUCATION": "education",
    "MARRIAGE": "marriage",
    "AGE": "age",
    "PAY_0": "pay_status_1",
    **{f"PAY_{m}": f"pay_status_{m}" for m in MONTHS[1:]},
    **{f"BILL_AMT{m}": f"bill_amt_{m}" for m in MONTHS},
    **{f"PAY_AMT{m}": f"pay_amt_{m}" for m in MONTHS},
    "default payment next month": TARGET,
}
