# ADR 001 — Pandera como contrato de dados, com portão bloqueante e quarentena

**Status:** aceito · **Data:** 2026-11-05

## Contexto

O enunciado exige um contrato de dados (Great Expectations ou Pandera) com pelo menos três regras rígidas e a demonstração de um lote ruim sendo bloqueado. O contrato precisa proteger dois pontos do fluxo: o treino (Dataset de Referência) e a escoragem (cada remessa de produção).

## Decisão

Usar **Pandera** (`pandera.pandas`) com um `DataFrameSchema` construído em código (`src/creditwatch/contracts/credit_contract.py`) e um portão próprio (`gate.py`) que:

1. valida em modo *lazy*, coletando todas as falhas de uma vez;
2. agrupa as falhas por regra de um catálogo com identificador estável (`DQ-01` … `DQ-13`, `DQ-W01`);
3. se houver violação, move o lote para a quarentena, grava o relatório e levanta `DataContractError` — nada é escorado;
4. devolve código de saída 1 na CLI (`creditwatch.pipelines.validate`), para servir de portão em orquestrador ou CI.

O esquema é `strict=True`: colunas fora do contrato reprovam o lote.

## Alternativas consideradas

| Alternativa | Por que não |
|---|---|
| Great Expectations | Traz um ecossistema próprio (contexto, *datasources*, *checkpoints*, *data docs*) cujo custo de configuração não se paga em um contrato de um único tipo de lote. O Pandera expressa as mesmas regras em Python puro, é testável com `pytest` e versionado junto do código. |
| Validação manual com `assert`/pandas | Sem relatório estruturado por regra, sem coleta *lazy* e sem um esquema declarativo que sirva de documentação. |
| Descartar só as linhas inválidas (`drop_invalid_rows`) | Esconde o problema: um lote com 3% de limites nulos indica falha na origem, e escorar os 97% restantes produz uma amostra enviesada. O lote inteiro é bloqueado e a origem reenvia. |

## Consequências

- **Positivas:** cada regra é testada isoladamente por um injetor de defeito (`simulation/corruption.py`); o identificador `DQ-xx` aparece no relatório, no log, na métrica Prometheus e no dashboard; `strict=True` implementa minimização de dados por padrão (uma coluna `cpf` enviada por engano nunca chega ao modelo).
- **Negativas:** bloquear o lote inteiro atrasa decisões de crédito válidas até o reenvio; a quarentena guarda o lote bruto e por isso tem retenção curta e acesso restrito (README, plano de retenção).
- **Limite deliberado:** o contrato trata de valores *inválidos*. Mudança de *distribuição* de valores válidos não reprova o lote; é assunto da detecção de drift ([ADR 002](002-psi-como-criterio-de-drift.md)).
