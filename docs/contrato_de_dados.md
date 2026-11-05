# Contrato de dados e portão de qualidade (Etapa 1)

O contrato de dados é a primeira barreira do pipeline: nenhum lote chega ao treino ou à escoragem sem passar por ele. Código em [`src/creditwatch/contracts/`](../src/creditwatch/contracts/).

## 1. Onde o contrato atua

```
 arquivo bruto (UCI) ──▶ contrato ──✗ REPROVA (DQ-05)      evidência de que o dado real chega sujo
        │
        ▼ limpeza documentada
 dataset limpo ───────▶ contrato ──✓ portão bloqueante ──▶ Dataset de Referência ──▶ treino

 remessa de produção ─▶ contrato ──✓ ──▶ escoragem ──▶ drift ──▶ performance
                            │
                            ✗ ──▶ quarentena + relatório + alerta crítico   (nada é escorado)
```

O mesmo esquema serve aos dois pontos; a única diferença é o rótulo: obrigatório na referência e **proibido** nas remessas de produção, porque no dia da decisão ele ainda não existe.

## 2. Catálogo de regras

Cada regra tem um identificador estável, usado no relatório de validação, nos logs, na métrica `creditwatch_contract_violations_total{rule=...}` e no dashboard. Definição em [`rules.py`](../src/creditwatch/contracts/rules.py) e [`credit_contract.py`](../src/creditwatch/contracts/credit_contract.py).

| Regra | Título | O que exige | Severidade |
|---|---|---|---|
| `DQ-01` | Completude | Nenhuma coluna obrigatória com valor nulo | bloqueante |
| `DQ-02` | Idade válida | `age` entre 18 e 100 (capacidade civil para contratar crédito) | bloqueante |
| `DQ-03` | Limite de crédito | `limit_bal` entre 1.000 e 1.000.000 (teto da política de crédito) | bloqueante |
| `DQ-04` | Unicidade | `customer_id` sem repetição dentro do lote | bloqueante |
| `DQ-05` | Domínio dos categóricos | `sex`, `education` e `marriage` restritos ao dicionário de dados | bloqueante |
| `DQ-06` | Status de pagamento | `pay_status_1..6` inteiros entre −2 e 8 | bloqueante |
| `DQ-07` | Pagamentos não negativos | `pay_amt_1..6` ≥ 0 | bloqueante |
| `DQ-08` | Faturas plausíveis | `bill_amt_1..6` entre −1.000.000 e 10.000.000 (erro de unidade ou escala) | bloqueante |
| `DQ-09` | Esquema estrito | Todas as colunas do contrato presentes e **nenhuma coluna extra** (ex.: PII) | bloqueante |
| `DQ-10` | Tipos | Colunas numéricas inteiras, sem texto ou fração | bloqueante |
| `DQ-11` | Volume mínimo | Pelo menos 500 linhas no lote (arquivo truncado) | bloqueante |
| `DQ-12` | Rótulo binário | Quando presente, `default` só admite 0 ou 1 | bloqueante |
| `DQ-13` | Mês de referência | `reference_month` no formato `AAAA-MM` e único no lote | bloqueante |
| `DQ-W01` | Carteira inativa | Menos de 60% dos clientes sem consumo no mês mais recente | aviso |

O enunciado pede ao menos três regras rígidas (sem nulos, faixa de idade, limites): são `DQ-01`, `DQ-02` e `DQ-03`, mais dez.

**Regra rígida × aviso × drift.** Regras bloqueantes tratam de valores *impossíveis ou inválidos*. `DQ-W01` é uma regra estatística de lote e apenas avisa. Mudança de distribuição de valores válidos (clientes mais jovens, faturas maiores) não é erro de dado: é tratada pela [detecção de drift](drift.md).

## 3. O que acontece quando um lote reprova

`enforce_contract` ([`gate.py`](../src/creditwatch/contracts/gate.py)) valida em modo *lazy* — coleta todas as falhas, não só a primeira — e, havendo violação:

1. copia o lote para `data/quarantine/<remessa>/batch.csv`, com o `validation_report.json` ao lado;
2. grava o relatório em `reports/data_quality/blocked_<remessa>.json`;
3. levanta `DataContractError`; no pipeline de monitoramento o estágio `contract` é marcado como `blocked`, um alerta crítico `data_quality` é emitido e os estágios seguintes **não rodam** para aquela remessa.

O lote inteiro é bloqueado, e não só as linhas inválidas: 3% de limites nulos indicam falha na origem, e escorar os 97% restantes produziria uma amostra enviesada sem que ninguém notasse ([ADR 001](adr/001-pandera-como-contrato-de-dados.md)).

**Privacidade do relatório de falha.** Exemplos de valores inválidos só são registrados para colunas previstas no contrato (ex.: idades `13`, `176`). De colunas inesperadas fica apenas o nome, e do identificador do cliente nunca se registra valor. Um teste garante que nenhum CPF do lote de demonstração aparece no relatório serializado.

## 4. Demonstração: o lote com erros

### 4.1 O dado real já chega sujo

O arquivo original da UCI **reprova** no contrato, sem nenhuma manipulação:

```json
{
  "batch": "raw", "status": "bloqueado", "n_rows": 30000, "rows_with_failures": 399,
  "violations": [{
    "rule_id": "DQ-05", "title": "Domínio dos categóricos",
    "columns": ["education", "marriage"], "n_failures": 399, "examples": ["5", "6", "0"]
  }]
}
```

São 345 linhas com `education` em códigos que o dicionário não documenta (0, 5, 6) e 54 com `marriage = 0`. A limpeza ([`cleaning.py`](../src/creditwatch/data/cleaning.py)) recodifica esses valores para "outros" e remove 35 clientes duplicados; só o resultado (29.965 linhas) passa no portão e vira Dataset de Referência. Relatórios: [`raw_validation.json`](../reports/data_quality/raw_validation.json), [`ingestion_report.json`](../reports/data_quality/ingestion_report.json).

### 4.2 Remessa corrompida pelo sistema de origem

Em 2025-07 a primeira remessa do mês chega com sete defeitos injetados por [`corruption.py`](../src/creditwatch/simulation/corruption.py). Para reproduzir:

```bash
make demo-bad-batch
# ou: poetry run python -m creditwatch.pipelines.validate data/production/batch_2025-07_remessa1_corrompida.parquet
```

```
Contrato: credit_batch | Lote: batch_2025-07_remessa1_corrompida | 2040 linhas x 26 colunas
Resultado: BLOQUEADO
Linhas com pelo menos uma falha: 246

Regra    Falhas  Titulo                     Colunas
DQ-01        60  Completude                 limit_bal
DQ-02        40  Idade válida               age
DQ-03        20  Limite de crédito          limit_bal
DQ-04        80  Unicidade                  customer_id
DQ-05        40  Domínio dos categóricos    education, sex
DQ-07        20  Pagamentos não negativos   pay_amt_1
DQ-09         1  Esquema estrito            cpf
ERROR creditwatch.pipelines.validate: ingestao BLOQUEADA: DQ-01, DQ-02, DQ-03, DQ-04, DQ-05, DQ-07, DQ-09
```

O comando termina com **código de saída 1**, o que permite usá-lo como portão em um orquestrador ou no CI (o workflow faz exatamente esse teste). No monitoramento, a remessa `2025-07-r1` aparece como bloqueada, com 2.040 linhas em quarentena e nenhuma escorada; a remessa corrigida `2025-07-r2` é processada normalmente. Relatório: [`blocked_2025-07-r1.json`](../reports/data_quality/blocked_2025-07-r1.json).

| Defeito injetado | Incidente que reproduz | Regra |
|---|---|---|
| Limite nulo em 3% das linhas | Falha de integração com o sistema de limites | `DQ-01` |
| Idades de 12 a 17 e de 120 a 199 | Erro de digitação ou de formulário | `DQ-02` |
| Limite multiplicado por 100 | Valor em centavos em vez de unidades | `DQ-03` |
| 2% das propostas repetidas | Reenvio parcial do arquivo | `DQ-04` |
| `sex = 0`, `education = 9` | Códigos fora do dicionário | `DQ-05` |
| Pagamento negativo | Estorno lançado como pagamento | `DQ-07` |
| Coluna `cpf` | A origem passa a enviar um dado pessoal que o modelo não usa | `DQ-09` |

Os CPFs da demonstração são fictícios e **inválidos por construção** (o dígito verificador é deslocado), para que o exemplo nunca reproduza o documento de uma pessoa real.

## 5. Como o contrato é testado

[`tests/test_contract.py`](../tests/test_contract.py) tem um teste parametrizado por defeito: cada um dos doze injetores precisa violar **exatamente** a sua regra e nenhuma outra. Há ainda testes para o dado bruto (399 falhas em `DQ-05`), para o rótulo proibido em produção, para a coerção de tipos, para a quarentena e para a não exposição de PII no relatório.
