# Simulação e detecção de drift (Etapa 2)

## 1. Referência × produção

| Dataset | Origem | Linhas | Uso |
|---|---|---|---|
| **Referência** (`data/reference/reference.parquet`) | Metade do dataset limpo, estratificada pelo rótulo | 14.982 (treino 8.989 · validação 2.996 · teste 2.997) | Única base de treino e avaliação do modelo; linha de base de todos os testes de drift |
| **Produção** (`data/production/`) | Doze remessas de 2.000 propostas, amostradas da outra metade (14.983 clientes nunca vistos no treino) | 24.000 escoradas | Onde o drift acontece |

Cada remessa chega em dois arquivos, como em produção: `batch_<mês>.parquet` com as propostas **sem rótulo** e `labels_<mês>.parquet` com a inadimplência observada, que só fica disponível depois da maturação da safra (2 meses, `CREDITWATCH_LABEL_LAG_MONTHS`).

As doze remessas são sorteadas do mesmo conjunto de 14.983 clientes, então **o mesmo cliente pode aparecer em mais de um mês** (com outro identificador de proposta). Para o monitoramento isso é indiferente: cada mês é comparado com a referência. Para experimentos que treinam em uns meses e testam em outros, não é — por isso o backtest de remediação usa clientes diferentes de cada lado ([README, seção 10.4](../README.md#104-o-que-a-remediação-consegue-backtest)).

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
| `stress` | drift de dados | Cresce a parcela de clientes com atraso recente (aperto financeiro) | Reamostragem com pesos (*exponential tilting*) sobre o atraso recente |
| `new_profiles` | drift de dados | A carteira recebe clientes mais jovens e com limite menor | Reamostragem com pesos sobre idade e limite |
| `concept` | drift de conceito | Clientes em dia e com utilização do limite a partir de 50% passam a inadimplir; parte dos inadimplentes é curada por renegociação | Alteração do rótulo; **nenhuma feature muda** |

Os três primeiros mudam o que o modelo *vê*; o quarto muda o que os mesmos dados *significam*. Duas ressalvas sobre a fronteira entre os dois tipos:

- a reamostragem não altera o rótulo de nenhum cliente, mas `new_profiles` pondera por idade, que não é entrada do modelo; a relação entre as features do modelo e a inadimplência pode variar um pouco com a composição da carteira;
- a inflação muda os valores e mantém o rótulo. Medida em valores nominais, a relação entre features e inadimplência se desloca: o mesmo risco passa a aparecer com faturas maiores.

Na análise causal, esses dois efeitos somam −0,029 de ROC AUC, contra −0,162 do drift de conceito.

**Reprodutibilidade e comparação entre cenários.** Cada remessa usa uma seed derivada da seed global e do mês. A amostragem usa o truque de Gumbel (somar um ruído ao log do peso de cada cliente e ficar com os maiores) e o sorteio do drift de conceito pertence ao cliente, não à posição no lote. Com a mesma seed, dois cenários que diferem em um mecanismo compartilham todo o ruído dos demais — é o que permite ligar e desligar mecanismos e comparar os resultados.

## 3. Como o drift é detectado

Para cada remessa, [`detector.py`](../src/creditwatch/drift/detector.py) compara 30 features com a referência: as 26 entradas do modelo e os 4 atributos sociodemográficos (monitorados como perfil da carteira, embora fora do modelo).

| Estatística | Onde | Papel |
|---|---|---|
| **PSI** (*Population Stability Index*) | todas as features | Tamanho do efeito; decide a severidade |
| **Kolmogorov-Smirnov** (D e p-valor) | 19 numéricas | Evidência; eleva de estável para atenção se D ≥ 0,10 e p < 0,05 |
| **Qui-quadrado** (p-valor) com **distância de variação total** | 11 categóricas | Evidência; mesmo critério com variação total ≥ 0,10 |

| Situação | Critério |
|---|---|
| Estável | PSI < 0,10 e efeito do teste < 0,10 |
| Atenção | 0,10 ≤ PSI < 0,25, ou efeito ≥ 0,10 com p < 0,05 |
| Crítico | PSI ≥ 0,25 |
| **Drift do dataset** | 30% ou mais das features em atenção ou crítico |

**Por que não decidir por p-valor.** Com 2.000 propostas contra 15.000 de referência, o p-valor acusa diferenças irrelevantes: em dezembro, 28 das 30 features têm p < 0,001, inclusive as que mal se moveram. O PSI mede o *quanto* mudou ([ADR 002](adr/002-psi-como-criterio-de-drift.md)).

**Tamanhos de efeito que não dependem do tamanho do lote.** O D do KS é a maior distância entre as distribuições acumuladas. A distância de variação total é a fração da carteira que mudou de categoria (metade da soma das diferenças absolutas entre as proporções). Uma mesma mudança de proporção dá o mesmo valor com lotes de 1.000 ou de 30.000 linhas — o que não acontece com o V de Cramér, que depende da razão entre os tamanhos das amostras.

O PSI numérico usa os decis da referência como faixas, fundindo bordas repetidas — relevante para variáveis com massa em zero, como os pagamentos. Implementação e testes: [`statistics.py`](../src/creditwatch/drift/statistics.py), [`tests/test_statistics.py`](../tests/test_statistics.py).

Além das entradas, o monitoramento acompanha o **drift de predição** (PSI e KS do score contra o score da referência fora do treino) e, quando o rótulo amadurece, o **drift do alvo** (teste z de duas proporções para a taxa de inadimplência).

## 4. Evidently

O [Evidently](https://www.evidentlyai.com/) compara a referência com a remessa e gera o relatório HTML navegável: distribuições lado a lado, teste por coluna, fração de colunas com drift e, quando há rótulo, a qualidade da classificação (`DataDriftPreset` com método PSI + `ClassificationPreset`). Código: [`evidently_report.py`](../src/creditwatch/drift/evidently_report.py).

Por padrão o relatório é gerado no último mês de cada fase (*checkpoints*); `make evidently-all` gera os doze.

| Checkpoint | Relatório | Colunas com drift (Evidently) | Features com drift (detector próprio) |
|---|---|---|---|
| mar/2025 — fim da fase estável | [`drift_2025-03.html`](../reports/evidently/drift_2025-03.html) | 0 de 31 | 0 de 30 |
| ago/2025 — fim do drift de dados | [`drift_2025-08.html`](../reports/evidently/drift_2025-08.html) | 16 de 31 (52%) | 22 de 30 (73%) |
| dez/2025 — fim do drift de conceito | [`drift_2025-12.html`](../reports/evidently/drift_2025-12.html) | 16 de 31 (52%) | 18 de 30 (60%) |

**Os dois não chegam ao mesmo número, e não deveriam.** O Evidently analisa 31 colunas porque inclui o score. E os valores de PSI diferem por dois motivos:

1. **Faixas.** O detector próprio usa os decis da referência; o Evidently monta as faixas de outra forma. Em dezembro, `utilization_1` tem PSI 0,81 no detector próprio e 1,67 no Evidently; `pay_amt_1`, 0,11 contra 0,01.
2. **Base de comparação.** O Evidently compara com as partições de validação e teste da referência (cerca de 6 mil clientes fora do treino), porque o relatório também traz o score e a qualidade da classificação, que seriam otimistas nas linhas de treino. O detector próprio compara as features com a referência inteira (cerca de 15 mil).

O detector próprio aponta algumas features a mais: valores brutos de fatura e de pagamento que cruzam um limiar por pouco, pelo PSI com decis ou pelo tamanho do efeito do KS. O que se exige é que os dois concordem sobre *quais* variáveis mudaram: nos três checkpoints, tudo o que o Evidently aponta o detector próprio também aponta, e nenhum dos dois acusa nada no mês estável. Essa contraprova é um teste automatizado ([`tests/test_integrations.py`](../tests/test_integrations.py)).

## 5. Resultado: o que mudou e quando

Série completa em [`timeline.csv`](../reports/monitoring/timeline.csv); por feature, em [`feature_drift.csv`](../reports/monitoring/feature_drift.csv).

| Mês | Fase | Features com drift | Críticas | PSI do score | Aprovação | ROC AUC | Inadimplência | Inadimpl. entre aprovados |
|---|---|---|---|---|---|---|---|---|
| jan | Estável | 0 | 0 | 0,010 | 75,7% | 0,773 | 23,0% | 13,5% |
| fev | Estável | 0 | 0 | 0,008 | 76,6% | 0,772 | 21,8% | 13,0% |
| mar | Estável | 0 | 0 | 0,003 | 75,4% | 0,778 | 21,8% | 12,3% |
| abr | Drift de dados | 3 | 0 | 0,006 | 72,8% | 0,785 | 24,2% | 13,4% |
| mai | Drift de dados | 4 | 2 | 0,024 | 70,6% | 0,774 | 24,2% | 13,1% |
| jun | Drift de dados | 9 | 3 | 0,067 | 66,2% | 0,792 | 29,2% | 14,6% |
| jul | Drift de dados | 13 | 5 | 0,146 | 59,8% | 0,812 | 31,0% | 14,0% |
| ago | Drift de dados | 22 | 10 | 0,278 | 55,1% | 0,799 | 34,8% | 15,3% |
| set | Drift de conceito | 20 | 10 | 0,257 | 55,0% | 0,755 | 36,2% | 20,2% |
| out | Drift de conceito | 20 | 10 | 0,271 | 54,6% | 0,696 | 36,2% | 23,6% |
| nov | Drift de conceito | 19 | 9 | 0,235 | 57,7% | 0,637 | 37,0% | 29,8% |
| dez | Drift de conceito | 18 | 9 | 0,217 | 57,6% | 0,595 | 36,0% | 31,0% |

Referência (partição de teste, 2.997 clientes): ROC AUC 0,780 · aprovação 74,4% · inadimplência 22,1% · inadimplência entre aprovados 12,0%.

### Features em drift em dezembro (18 de 30)

| Feature | Tipo | PSI | Teste | Tamanho do efeito | Situação |
|---|---|---|---|---|---|
| `utilization_1` | numérica | 0,813 | KS | 0,413 | crítico |
| `bill_trend` | numérica | 0,605 | KS | 0,331 | crítico |
| `payment_ratio_1` | numérica | 0,504 | KS | 0,248 | crítico |
| `utilization_avg` | numérica | 0,491 | KS | 0,303 | crítico |
| `payment_ratio_avg` | numérica | 0,392 | KS | 0,195 | crítico |
| `pay_status_2` | categórica | 0,274 | qui² | 0,213 | crítico |
| `pay_status_1` | categórica | 0,263 | qui² | 0,218 | crítico |
| `months_delayed` | categórica | 0,261 | qui² | 0,212 | crítico |
| `max_delay` | categórica | 0,250 | qui² | 0,213 | crítico |
| `limit_bal` | numérica | 0,231 | KS | 0,209 | atenção |
| `pay_status_3` | categórica | 0,183 | qui² | 0,158 | atenção |
| `age` (fora do modelo) | numérica | 0,153 | KS | 0,148 | atenção |
| `pay_status_4` | categórica | 0,148 | qui² | 0,129 | atenção |
| `pay_status_5` | categórica | 0,126 | qui² | 0,114 | atenção |
| `pay_amt_1` | numérica | 0,114 | KS | 0,103 | atenção |
| `pay_status_6` | categórica | 0,113 | qui² | 0,102 | atenção |
| `bill_amt_1` | numérica | 0,063 | KS | 0,110 | atenção |
| `bill_amt_2` | numérica | 0,055 | KS | 0,106 | atenção |

`bill_amt_1` e `bill_amt_2` têm PSI abaixo de 0,10 e entram em atenção pelo tamanho do efeito do KS. `sex`, `education` e `marriage` permanecem estáveis.

## 6. Leituras

1. **Nenhum alarme falso.** Nos três meses de controle, nenhuma feature passa de PSI 0,012 e nenhum alerta é emitido.
2. **O drift de dados é detectado cedo.** O primeiro alerta sai em abril, com 3 features em atenção; em junho já há drift do dataset (30% das features).
3. **No drift de dados, a ordenação se mantém e o resto muda.** De abril a agosto, 22 features mudam e o ROC AUC fica entre 0,77 e 0,81: o modelo continua ordenando bem uma população que ficou mais arriscada. Mas a aprovação cai de 76% para 55%, a inadimplência entre aprovados sobe de 12,0% para 15,3% e o erro de calibração do segmento em atraso chega a 4,7 p.p. Drift de dados não derrubou este modelo, e também não foi inofensivo.
4. **O drift de conceito não aparece nos testes de entrada.** De agosto para setembro o número de features com drift até diminui (22 → 20), e é justamente aí que o AUC começa a cair. Só as métricas com rótulo o revelam, e com dois meses de atraso.
5. **O alerta de drift de dados não antecipa a queda.** O primeiro alerta é de abril e a queda começa em setembro, mas por outro mecanismo. A sequência é do cenário, não uma propriedade do detector.
6. **Valores monetários brutos são pouco sensíveis à inflação.** Faturas 70% maiores movem o PSI de `bill_amt_1` para apenas 0,06, porque a distribuição é muito dispersa; as razões (`utilization_1`, PSI 0,81) captam a mudança muito antes. Monitorar só as colunas brutas atrasaria o alerta.

O que fazer diante de cada alerta está no [runbook](runbook.md).
