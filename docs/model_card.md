# Model Card — modelo baseline de risco de crédito

## Identificação

| | |
|---|---|
| Nome | `credit-scoring-baseline` |
| Tipo | XGBoost (`XGBClassifier`), classificação binária |
| Saída | Probabilidade de inadimplência no mês seguinte (PD) |
| Decisão | Aprovar se PD < 0,289 (ponto de corte que maximiza o KS na validação) |
| Versão | Registrada em [`models/metadata.json`](../models/metadata.json) (`AAAAMMDD-<hash da referência>`) |
| Código | [`src/creditwatch/model/`](../src/creditwatch/model/) |
| Rastreabilidade | `metadata.json` guarda hiperparâmetros, métricas, versões das bibliotecas e o SHA-256 do dataset de referência e do artefato |

## Uso pretendido

- **Para que serve:** ordenar clientes de cartão de crédito por risco de inadimplência e apoiar a decisão de concessão ou manutenção de limite.
- **Para que não serve:** decidir sozinho, sem canal de revisão humana; ser aplicado a outro produto, país ou período sem revalidação; servir de base para precificação individual.
- **Contexto deste repositório:** o modelo é propositalmente simples. O objeto do projeto é a camada de sustentação — contrato de dados, detecção de drift, observabilidade e governança — e não a maximização de AUC.

## Dados

- **Fonte:** *Default of Credit Card Clients* (UCI, CC BY 4.0): 30.000 clientes de um banco de Taiwan, abril a setembro de 2005. Ver [`data/README.md`](../data/README.md).
- **Preparação:** códigos de `education` e `marriage` fora do dicionário recodificados para "outros" (399 linhas) e 35 duplicatas removidas; 29.965 linhas passam no contrato de dados.
- **Referência:** 14.982 clientes (treino 8.989 · validação 2.996 · teste 2.997), 22,1% de inadimplência. A outra metade é reservada para a simulação da produção e nunca entra no treino.

## Features

26 entradas, todas de comportamento financeiro:

| Grupo | Features |
|---|---|
| Limite | `limit_bal` |
| Situação de pagamento (6 meses) | `pay_status_1..6` |
| Faturas (6 meses) | `bill_amt_1..6` |
| Pagamentos (6 meses) | `pay_amt_1..6` |
| Derivadas | `utilization_1`, `utilization_avg`, `payment_ratio_1`, `payment_ratio_avg`, `max_delay`, `months_delayed`, `bill_trend` |

**Fora do modelo por decisão de governança:** `sex`, `age`, `marriage`, `education` ([ADR 004](adr/004-atributos-protegidos-fora-do-modelo.md)). Incluí-los não melhora o desempenho: ROC AUC de teste 0,7806 com eles contra 0,7809 sem (três seeds).

Importância (ganho): `max_delay` 26,5%, `pay_status_1` 19,0%, `months_delayed` 16,7%. O modelo depende sobretudo do histórico de atrasos — é por isso que o drift de conceito, que quebra a relação entre "estar em dia" e "pagar", o atinge tanto.

## Desempenho na referência

| Candidato | ROC AUC (validação) | KS (validação) |
|---|---|---|
| Regressão Logística | 0,760 | 0,400 |
| **XGBoost** (selecionado) | **0,774** | **0,423** |

| Métrica (teste, 2.997 clientes) | Valor |
|---|---|
| ROC AUC | 0,780 |
| Gini | 0,560 |
| KS | 0,445 |
| Brier | 0,135 |
| Precisão / recall da classe inadimplente | 0,515 / 0,596 |
| Diferença de calibração (observada − prevista) | −0,5 p.p. |
| Taxa de aprovação | 74,4% |
| Inadimplência entre aprovados | 12,0% |

Hiperparâmetros: 300 árvores, profundidade 4, taxa de aprendizado 0,05, `subsample` e `colsample_bytree` 0,8, `min_child_weight` 5, seed 42.

## Desempenho em produção (cenário simulado)

| Fase | Meses | ROC AUC | Inadimplência entre aprovados |
|---|---|---|---|
| Estável | jan–mar | 0,773 a 0,786 | 12,2% a 12,4% |
| Drift de dados | abr–ago | 0,775 a 0,813 | 12,7% a 15,8% |
| Drift de conceito | set–dez | 0,741 → 0,613 | 19,7% → 30,7% |

O modelo resiste ao drift de dados e degrada com o drift de conceito. Detalhes em [drift.md](drift.md) e na análise causal do README.

## Equidade

Auditoria por sexo, faixa etária, escolaridade e estado civil a cada execução ([`fairness_audit.json`](../reports/analysis/fairness_audit.json)). Razão de impacto desigual do grupo menos aprovado (1,00 = igual ao grupo mais aprovado; abaixo de 0,80 indica impacto adverso):

| Atributo | Referência | Estável | Drift de dados | Drift de conceito |
|---|---|---|---|---|
| Sexo | 0,94 (masculino) | 0,96 | 0,94 | 0,91 |
| Faixa etária | 0,95 (18–29) | 0,93 | 0,91 | 0,84 (50+) |
| Escolaridade | 0,87 (ensino médio) | 0,86 | **0,83** (ensino médio) | 0,84 (graduação) |
| Estado civil | 0,98 (casado) | 0,99 | 1,00 | 0,96 |

Nenhum grupo fica abaixo do limite em nenhum período, mas a disparidade por escolaridade existe desde a referência (0,87) e aumenta com o drift (0,83): tirar o atributo das entradas não impede que as variáveis financeiras o reflitam. Entre bons pagadores, a taxa de aprovação do grupo menos favorecido chega a ficar 10,7 pontos percentuais abaixo da do mais favorecido (graduação, no drift de conceito).

Grupos com menos de 100 clientes ou de 2% da amostra do período não entram na comparação (a categoria "outros" de escolaridade e de estado civil), porque a taxa de aprovação de um grupo pequeno é ruidosa demais para servir de referência.

## Limitações e riscos

- **Dados antigos e de outro país.** Clientes de Taiwan em 2005; a relação entre comportamento e inadimplência não se transfere a outra carteira.
- **Rótulo de curto prazo.** Inadimplência no mês seguinte; modelos de concessão costumam usar janelas de 6 a 12 meses.
- **Viés de seleção.** A base só contém clientes já aprovados; o modelo não foi validado em quem foi negado.
- **Drift simulado.** A produção é gerada por um cenário controlado. A detecção foi validada contra mecanismos conhecidos; em produção, as causas reais precisam ser investigadas ([runbook](runbook.md)).
- **Dependência do histórico de atrasos.** Três features concentram 62% da importância; uma mudança nessa relação derruba o modelo.
- **Atraso de rótulo.** A degradação só é mensurável dois meses depois da decisão.

## Manutenção

- Monitoramento a cada remessa: contrato, drift, score, performance e viés ([observabilidade.md](observabilidade.md)).
- Critério de revisão: alerta crítico de performance (queda de ROC AUC ≥ 0,05) ou impacto adverso na auditoria de viés.
- Retreino: avaliado como desafiante em meses posteriores ao treino antes de qualquer promoção ([runbook](runbook.md), seção 4).
