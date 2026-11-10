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
- **Para que não serve:** decidir sozinho, sem canal de revisão; ser aplicado a outro produto, país ou período sem revalidação; servir de base para precificação individual.
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

**Fora do modelo por decisão de governança:** `sex`, `age`, `marriage`, `education` ([ADR 004](adr/004-atributos-protegidos-fora-do-modelo.md)). Incluí-los não melhora o desempenho: treinando o mesmo algoritmo em três seeds, o ROC AUC de teste é 0,7806 com eles e 0,7809 sem (na validação, 0,7782 e 0,7754).

Importância (ganho): `max_delay` 26,5%, `pay_status_1` 19,0%, `months_delayed` 16,7%. O modelo depende sobretudo do histórico de atrasos — é por isso que o drift de conceito, que quebra a relação entre "estar em dia" e "pagar", o atinge tanto.

## Desempenho na referência

| Candidato | ROC AUC (validação) | KS (validação) |
|---|---|---|
| Regressão Logística | 0,760 | 0,400 |
| **XGBoost** (selecionado) | **0,774** | **0,423** |

A validação serve para escolher o modelo e o ponto de corte. As métricas abaixo são da partição de teste, que não participou de nenhuma escolha, e são a base de comparação de todo o monitoramento.

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
| Estável | jan–mar | 0,772 a 0,778 | 12,3% a 13,5% |
| Drift de dados | abr–ago | 0,774 a 0,812 | 13,1% a 15,3% |
| Drift de conceito | set–dez | 0,755 → 0,595 | 20,2% → 31,0% |

No drift de dados a ordenação se mantém e a inadimplência entre aprovados sobe 3 pontos; no drift de conceito o modelo perde a maior parte do poder de separação. Detalhes em [drift.md](drift.md) e na análise causal do README.

**Retreinar não devolve o desempenho.** No backtest sem vazamento ([README, seção 10.4](../README.md#104-o-que-a-remediação-consegue-backtest)), a melhor estratégia com os rótulos disponíveis leva o ROC AUC de 0,610 para 0,631, e um retreino com rótulos imediatos chegaria a 0,646.

## Equidade

Auditoria por sexo, faixa etária, escolaridade e estado civil a cada execução ([`fairness_audit.json`](../reports/analysis/fairness_audit.json)). Razão de impacto desigual do grupo menos aprovado, com intervalo de 95% (1,00 = igual ao grupo mais aprovado; abaixo de 0,80 indica impacto adverso pela regra dos quatro quintos, usada aqui como critério de triagem):

| Atributo | Referência | Drift de dados | Drift de conceito |
|---|---|---|---|
| Sexo (masculino) | 0,94 (0,91 a 0,97) | 0,95 (0,92 a 0,98) | 0,91 (0,88 a 0,95) |
| Faixa etária | 0,95 (0,92 a 0,98), 18–29 | 0,91 (0,88 a 0,94), 18–29 | 0,91 (0,82 a 1,01), 50+ |
| Escolaridade (ensino médio) | 0,87 (0,83 a 0,92) | **0,83 (0,80 a 0,87)** | **0,82 (0,77 a 0,87)** |
| Estado civil | 0,98 (0,95 a 1,01) | 1,00 (0,97 a 1,03) | 1,00 (0,96 a 1,04) |

- Nenhum grupo fica abaixo de 0,80 na estimativa pontual; para escolaridade, o intervalo cruza o limite nos dois períodos de drift e o resultado é inconclusivo.
- Entre bons pagadores, o grupo "ensino médio" é aprovado 9,0 p.p. menos que o grupo mais favorecido na referência e 13,2 p.p. menos no drift de conceito. Tirar o atributo das entradas não impede que as variáveis financeiras o reflitam.
- Grupos com menos de 100 clientes ou de 2% da amostra do período (a categoria "outros" de escolaridade e de estado civil) aparecem no relatório, marcados como fora da comparação: a taxa de um grupo pequeno é ruidosa demais para servir de referência.
- Nenhuma técnica de mitigação foi aplicada; o plano está no README, seção 9.7.

## Limitações e riscos

- **Dados antigos e de outro país.** Clientes de Taiwan em 2005; a relação entre comportamento e inadimplência não se transfere a outra carteira.
- **Rótulo de curto prazo.** Inadimplência no mês seguinte; modelos de concessão costumam usar janelas de 6 a 12 meses.
- **Viés de seleção.** A base só contém clientes que já têm cartão; o modelo não foi validado em quem foi negado.
- **Drift simulado.** A produção é gerada por um cenário controlado, e as doze remessas são sorteadas do mesmo conjunto de clientes. A detecção foi validada contra mecanismos conhecidos; em produção, as causas reais precisam ser investigadas ([runbook](runbook.md)).
- **Dependência do histórico de atrasos.** Três features concentram 62% da importância; uma mudança nessa relação derruba o modelo.
- **Atraso de rótulo.** A degradação só é mensurável dois meses depois da decisão, e é esse atraso que limita qualquer remediação por modelo.
- **Sem explicação por decisão.** A importância medida é global; informar ao titular os fatores de uma negativa específica não está implementado.

## Manutenção

- Monitoramento a cada remessa: contrato, drift, score, performance e viés ([observabilidade.md](observabilidade.md)).
- Critério de revisão: alerta crítico de performance (queda de ROC AUC ≥ 0,05) ou impacto adverso na auditoria de viés.
- Retreino: avaliado como desafiante em safras com rótulo maduro, sem clientes em comum com o teste, antes de qualquer promoção ([runbook](runbook.md), seção 4).
