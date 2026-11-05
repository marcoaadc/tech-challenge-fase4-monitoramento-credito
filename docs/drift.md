# Simulação e detecção de drift (Etapa 2)

## 1. Referência × produção

| Dataset | Origem | Linhas | Uso |
|---|---|---|---|
| **Referência** (`data/reference/reference.parquet`) | Metade do dataset limpo, estratificada pelo rótulo | 14.982 (treino 8.989 · validação 2.996 · teste 2.997) | Única base de treino e avaliação do modelo; linha de base de todos os testes de drift |
| **Produção** (`data/production/`) | Doze remessas de 2.000 propostas, amostradas da outra metade (14.983 clientes nunca vistos no treino) | 24.000 escoradas | Onde o drift acontece |

Cada remessa chega em dois arquivos, como em produção: `batch_<mês>.parquet` com as propostas **sem rótulo** e `labels_<mês>.parquet` com a inadimplência observada, que só fica disponível depois da maturação da safra (2 meses, `CREDITWATCH_LABEL_LAG_MONTHS`).

## 2. O cenário simulado

Narrativa: depois do treino, a economia sofre um choque inflacionário e a fintech passa a captar um público mais jovem; meses depois, o próprio comportamento de pagamento muda. Plano em [`scenario.py`](../src/creditwatch/simulation/scenario.py), mecanismos em [`mechanisms.py`](../src/creditwatch/simulation/mechanisms.py).

| Meses | Fase | `inflation` | `stress` | `new_profiles` | `concept` |
|---|---|---|---|---|---|
| jan–mar | Estável (controle) | 0 | 0 | 0 | 0 |
| abr–ago | Drift de dados | 0,10 → 0,50 | 0,10 → 0,50 | 0 → 0,75 (a partir de jun) | 0 |
| set–dez | Drift de conceito | 0,55 → 0,70 | 0,50 | 0,75 | 0,25 → 1,00 |

| Mecanismo | Tipo | O que muda | Como é implementado |
|---|---|---|---|
| `inflation` | drift de dados | As faturas sobem em termos nominais (até +70% no mês mais recente) e os pagamentos acompanham só 35% desse aumento | Multiplicação de `bill_amt_*` e `pay_amt_*`; limite e rótulo intactos |
| `stress` | drift de dados | Cresce a parcela de clientes com atraso recente (aperto financeiro) | Reamostragem com pesos (*exponential tilting*): muda P(x), preserva P(y\|x) |
| `new_profiles` | drift de dados | A carteira recebe clientes mais jovens e com limite menor | Reamostragem com pesos sobre idade e limite |
| `concept` | drift de conceito | Clientes em dia e alavancados (utilização ≥ 50%) passam a inadimplir; parte dos inadimplentes é curada por renegociação | Alteração do rótulo; **nenhuma feature muda** |

Os três primeiros mudam o que o modelo *vê*; o quarto muda o que os mesmos dados *significam*. Cada mecanismo pode ser ligado isoladamente, o que a [análise causal](../README.md#9-análise-causal-da-degradação) usa para atribuir a queda de desempenho.

A simulação é reprodutível: cada remessa usa uma seed derivada da seed global e do mês.

## 3. Como o drift é detectado

Para cada remessa, [`detector.py`](../src/creditwatch/drift/detector.py) compara 30 features com a referência: as 26 entradas do modelo e os 4 atributos sociodemográficos (monitorados como perfil da carteira, embora fora do modelo).

| Estatística | Onde | Papel |
|---|---|---|
| **PSI** (Population Stability Index) | todas as features | Tamanho do efeito; decide a severidade |
| **Kolmogorov-Smirnov** (D e p-valor) | 19 numéricas | Evidência; eleva de estável para atenção se D ≥ 0,10 e p < 0,05 |
| **Qui-quadrado** (V de Cramér e p-valor) | 11 categóricas | Evidência; mesmo critério com V ≥ 0,10 |

| Situação | Critério |
|---|---|
| Estável | PSI < 0,10 e efeito do teste < 0,10 |
| Atenção | 0,10 ≤ PSI < 0,25, ou efeito ≥ 0,10 com p < 0,05 |
| Crítico | PSI ≥ 0,25 |
| **Drift do dataset** | 30% ou mais das features em atenção ou crítico |

**Por que não decidir por p-valor.** Com 2.000 propostas contra 15.000 de referência, o p-valor acusa diferenças irrelevantes: em dezembro, 29 das 30 features têm p < 0,001, inclusive as que mal se moveram. O PSI mede o *quanto* mudou ([ADR 002](adr/002-psi-como-criterio-de-drift.md)).

O PSI numérico usa os decis da referência como faixas, fundindo bordas repetidas — relevante para variáveis com massa em zero, como os pagamentos. Implementação e testes: [`statistics.py`](../src/creditwatch/drift/statistics.py), [`tests/test_statistics.py`](../tests/test_statistics.py).

Além das entradas, o monitoramento acompanha o **drift de predição** (PSI e KS do score contra o score da referência fora do treino) e, quando o rótulo amadurece, o **drift do alvo** (teste z de duas proporções para a taxa de inadimplência).

## 4. Evidently

O [Evidently](https://www.evidentlyai.com/) compara a referência com a remessa e gera o relatório HTML navegável: distribuições lado a lado, teste por coluna, fração de colunas com drift e, quando há rótulo, a qualidade da classificação (`DataDriftPreset` com método PSI + `ClassificationPreset`). Código: [`evidently_report.py`](../src/creditwatch/drift/evidently_report.py).

Por padrão o relatório é gerado no último mês de cada fase (*checkpoints*); `make evidently-all` gera os doze.

| Checkpoint | Relatório | Colunas com drift (Evidently) | Features com drift (detector próprio) |
|---|---|---|---|
| mar/2025 — fim da fase estável | [`drift_2025-03.html`](../reports/evidently/drift_2025-03.html) | 0 de 31 | 0 de 30 |
| ago/2025 — fim do drift de dados | [`drift_2025-08.html`](../reports/evidently/drift_2025-08.html) | 16 de 31 (52%) | 23 de 30 (77%) |
| dez/2025 — fim do drift de conceito | [`drift_2025-12.html`](../reports/evidently/drift_2025-12.html) | 16 de 31 (52%) | 19 de 30 (63%) |

O Evidently analisa 31 colunas porque inclui o score. As duas contas concordam em tudo o que o Evidently aponta; o detector próprio marca algumas features a mais (valores brutos de fatura e de pagamento) porque também considera o tamanho do efeito do KS, não só o PSI. Não há nenhuma feature que só o Evidently aponte. Essa contraprova é um teste automatizado ([`tests/test_integrations.py`](../tests/test_integrations.py)).

## 5. Resultado: o que mudou e quando

Série completa em [`timeline.csv`](../reports/monitoring/timeline.csv); por feature, em [`feature_drift.csv`](../reports/monitoring/feature_drift.csv).

| Mês | Fase | Features com drift | Críticas | PSI do score | Aprovação | ROC AUC | Inadimplência | Inadimpl. entre aprovados |
|---|---|---|---|---|---|---|---|---|
| jan | Estável | 0 | 0 | 0,006 | 76,4% | 0,773 | 21,1% | 12,4% |
| fev | Estável | 0 | 0 | 0,010 | 76,0% | 0,786 | 21,8% | 12,2% |
| mar | Estável | 0 | 0 | 0,008 | 77,0% | 0,775 | 21,4% | 12,3% |
| abr | Drift de dados | 3 | 0 | 0,009 | 74,6% | 0,775 | 23,5% | 12,7% |
| mai | Drift de dados | 4 | 1 | 0,018 | 70,5% | 0,804 | 26,9% | 13,4% |
| jun | Drift de dados | 12 | 4 | 0,071 | 64,6% | 0,797 | 29,9% | 14,9% |
| jul | Drift de dados | 15 | 5 | 0,151 | 60,2% | 0,813 | 31,6% | 13,5% |
| ago | Drift de dados | 23 | 10 | 0,229 | 54,9% | 0,791 | 35,3% | 15,8% |
| set | Drift de conceito | 19 | 10 | 0,254 | 53,5% | 0,741 | 35,8% | 19,7% |
| out | Drift de conceito | 19 | 10 | 0,244 | 56,0% | 0,690 | 35,5% | 23,8% |
| nov | Drift de conceito | 20 | 10 | 0,247 | 55,0% | 0,651 | 37,1% | 28,6% |
| dez | Drift de conceito | 19 | 9 | 0,233 | 55,7% | 0,613 | 36,8% | 30,7% |

Referência (fora do treino): ROC AUC 0,777 · aprovação 74,9% · inadimplência 22,1% · inadimplência entre aprovados 12,3%.

### Features degradadas em dezembro (PSI contra a referência)

| Feature | Tipo | PSI | Teste | Estatística | Situação |
|---|---|---|---|---|---|
| `utilization_1` | numérica | 0,905 | KS | 0,441 | crítico |
| `bill_trend` | numérica | 0,658 | KS | 0,349 | crítico |
| `utilization_avg` | numérica | 0,570 | KS | 0,337 | crítico |
| `payment_ratio_1` | numérica | 0,519 | KS | 0,249 | crítico |
| `payment_ratio_avg` | numérica | 0,387 | KS | 0,189 | crítico |
| `pay_status_2` | categórica | 0,344 | qui² | 0,218 | crítico |
| `pay_status_1` | categórica | 0,302 | qui² | 0,198 | crítico |
| `months_delayed` | categórica | 0,291 | qui² | 0,194 | crítico |
| `max_delay` | categórica | 0,279 | qui² | 0,193 | crítico |
| `limit_bal` | numérica | 0,236 | KS | 0,206 | atenção |
| `age` (fora do modelo) | numérica | 0,127 | KS | 0,151 | atenção |

As demais em atenção são `pay_status_3..6`, `pay_amt_1` e `bill_amt_1..3`. `sex`, `education` e `marriage` permanecem estáveis.

## 6. Leituras

1. **Nenhum alarme falso.** Nos três meses de controle, nenhuma feature passa de PSI 0,02 e nenhum alerta é emitido.
2. **O drift de dados é detectado cedo.** O primeiro alerta sai em abril, com 3 features em atenção; em junho já há drift do dataset (40% das features).
3. **Drift de dados não é sinônimo de modelo ruim.** De abril a agosto, 23 features mudam e o ROC AUC *não cai* (0,77 a 0,81): a população ficou mais arriscada, mas o modelo continua ordenando bem. O que muda é o negócio — a aprovação cai de 76% para 55%.
4. **O drift de conceito não aparece nos testes de entrada.** De agosto para setembro o número de features com drift até diminui (23 → 19), e é justamente aí que o AUC começa a cair. Só as métricas com rótulo o revelam, e com dois meses de atraso.
5. **Valores monetários brutos são pouco sensíveis à inflação.** Faturas 70% maiores movem o PSI de `bill_amt_1` para apenas 0,09, porque a distribuição é muito dispersa; as razões (`utilization_1`, PSI 0,90) captam a mudança muito antes. Monitorar só as colunas brutas atrasaria o alerta.

O que fazer diante de cada alerta está no [runbook](runbook.md).
