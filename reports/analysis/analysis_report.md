# Análises de governança: causalidade, viés e remediação

Gerado por `python -m creditwatch.pipelines.analyze`. Os números mudam se o cenário ou a seed mudarem.

## 1. O que causou a degradação (atribuição por mecanismo)

Intervenções sobre o plano de 2025-12: as 16 combinações de mecanismos, em lotes de 2000 propostas (o tamanho das remessas de produção), repetidas com 5 seeds. Cada célula é a parcela do efeito total atribuída ao mecanismo (valor de Shapley, média ± desvio-padrão entre as seeds).

| Mecanismo | ROC AUC | Inadimplência entre aprovados | Taxa de aprovação | Features com drift |
|---|---|---|---|---|
| Inflação (faturas nominais) | -0.015 ± 0.005 | +0.020 ± 0.004 | +0.020 ± 0.005 | +6.2 ± 1.3 |
| Aperto financeiro (mais atrasos) | +0.022 ± 0.007 | +0.002 ± 0.002 | -0.178 ± 0.006 | +10.4 ± 0.3 |
| Novos perfis (jovens, limite baixo) | -0.014 ± 0.006 | +0.020 ± 0.003 | -0.041 ± 0.002 | +3.3 ± 0.6 |
| Drift de conceito (P(y|x)) | -0.162 ± 0.008 | +0.133 ± 0.005 | +0.000 ± 0.000 | +0.0 ± 0.0 |
| **Efeito total** | **-0.169** | **+0.175** | **-0.199** | **+20.0** |

Sem nenhum mecanismo: ROC AUC 0.774, 0.0 features com drift. Com todos: ROC AUC 0.606, 20.0 features com drift.

## 2. Onde procurar: varredura de calibração por feature

Para cada faixa de cada feature (categorias ou quartis da referência), o excesso de erro é a diferença entre inadimplência observada e prevista no período de drift de conceito, menos a mesma diferença na referência. A dispersão é o maior excesso menos o menor entre as faixas da feature: quanto maior, mais a feature separa quem o modelo passou a errar de quem ele continua acertando.

| Feature | Dispersão do excesso | Faixa com maior excesso | Excesso na faixa | Parcela da carteira |
|---|---|---|---|---|
| `pay_status_1` | 0.231 | 0 | +0.176 | 36.1% |
| `pay_status_2` | 0.196 | 0 | +0.160 | 41.0% |
| `months_delayed` | 0.177 | 0 | +0.118 | 44.6% |
| `bill_amt_1` | 0.172 | faixa 4 de 4 | +0.098 | 33.5% |
| `max_delay` | 0.160 | 0 | +0.118 | 44.6% |
| `pay_amt_1` | 0.160 | faixa 3 de 4 | +0.119 | 28.1% |
| `pay_status_3` | 0.155 | 0 | +0.121 | 46.5% |
| `utilization_1` | 0.144 | faixa 4 de 4 | +0.098 | 50.4% |

## 3. Inadimplência observada x prevista por segmento

Segmentos montados com as variáveis que a varredura aponta: situação do pagamento mais recente e utilização do limite, cortada na mediana da referência (0.34).

| Período | Segmento | Participação | Observada | Prevista | Diferença |
|---|---|---|---|---|---|
| Referência | em dia, utilização baixa | 40.0% | 13.9% | 14.0% | -0.1% |
| Referência | em dia, utilização alta | 37.2% | 13.9% | 13.7% | +0.2% |
| Referência | em atraso | 22.8% | 50.0% | 51.1% | -1.0% |
| Estável | em dia, utilização baixa | 41.3% | 12.5% | 13.6% | -1.1% |
| Estável | em dia, utilização alta | 36.5% | 16.0% | 13.6% | +2.4% |
| Estável | em atraso | 22.2% | 50.4% | 50.2% | +0.2% |
| Drift de dados | em dia, utilização baixa | 30.8% | 13.1% | 14.3% | -1.2% |
| Drift de dados | em dia, utilização alta | 33.6% | 15.1% | 14.9% | +0.2% |
| Drift de dados | em atraso | 35.6% | 54.9% | 50.2% | +4.7% |
| Drift de conceito | em dia, utilização baixa | 21.6% | 14.7% | 16.0% | -1.3% |
| Drift de conceito | em dia, utilização alta | 32.6% | 36.1% | 15.6% | +20.5% |
| Drift de conceito | em atraso | 45.9% | 46.8% | 50.0% | -3.2% |

## 4. Features candidatas a causa do drift de predição (PSI x importância)

| Feature | PSI | Importância | Impacto |
|---|---|---|---|
| `max_delay` | 0.250 | 0.265 | 0.0664 |
| `pay_status_1` | 0.263 | 0.190 | 0.0500 |
| `months_delayed` | 0.261 | 0.167 | 0.0434 |
| `utilization_1` | 0.813 | 0.018 | 0.0144 |
| `utilization_avg` | 0.491 | 0.019 | 0.0093 |
| `bill_trend` | 0.605 | 0.014 | 0.0087 |
| `pay_status_2` | 0.274 | 0.031 | 0.0085 |
| `payment_ratio_1` | 0.504 | 0.016 | 0.0078 |
| `payment_ratio_avg` | 0.392 | 0.014 | 0.0055 |
| `pay_status_3` | 0.183 | 0.023 | 0.0042 |

## 5. Auditoria de viés (grupo menos aprovado de cada atributo)

Impacto desigual: taxa de aprovação do grupo dividida pela do grupo mais aprovado, com intervalo de 95%. Abaixo de 0,80 indica impacto adverso (regra dos quatro quintos); se só o intervalo cruza 0,80, o resultado é inconclusivo. Diferença de oportunidade: quanto a taxa de aprovação entre bons pagadores do grupo fica abaixo da do grupo mais favorecido.

| Período | Atributo | Grupo | n | Impacto desigual (IC 95%) | Diferença de oportunidade | Situação |
|---|---|---|---|---|---|---|
| Referência | Sexo | masculino | 2400 | 0.94 (0.91 a 0.97) | +1.8 p.p. | dentro do limite |
| Referência | Faixa etária | 18-29 | 2009 | 0.95 (0.92 a 0.98) | +2.8 p.p. | dentro do limite |
| Referência | Escolaridade | ensino médio | 985 | 0.87 (0.83 a 0.92) | +9.0 p.p. | dentro do limite |
| Referência | Estado civil | casado(a) | 2684 | 0.98 (0.95 a 1.01) | +1.1 p.p. | dentro do limite |
| Estável | Sexo | masculino | 2311 | 0.95 (0.93 a 0.98) | +2.3 p.p. | dentro do limite |
| Estável | Faixa etária | 50+ | 543 | 0.93 (0.88 a 0.98) | +2.5 p.p. | dentro do limite |
| Estável | Escolaridade | graduação | 2783 | 0.90 (0.88 a 0.93) | +5.0 p.p. | dentro do limite |
| Estável | Estado civil | casado(a) | 2781 | 0.96 (0.93 a 0.99) | +1.2 p.p. | dentro do limite |
| Drift de dados | Sexo | masculino | 4044 | 0.95 (0.92 a 0.98) | +2.7 p.p. | dentro do limite |
| Drift de dados | Faixa etária | 18-29 | 3759 | 0.91 (0.88 a 0.94) | +3.8 p.p. | dentro do limite |
| Drift de dados | Escolaridade | ensino médio | 1643 | 0.83 (0.80 a 0.87) | +9.8 p.p. | inconclusivo |
| Drift de dados | Estado civil | casado(a) | 4286 | 1.00 (0.97 a 1.03) | +0.0 p.p. | dentro do limite |
| Drift de conceito | Sexo | masculino | 3316 | 0.91 (0.88 a 0.95) | +5.1 p.p. | dentro do limite |
| Drift de conceito | Faixa etária | 50+ | 330 | 0.91 (0.82 a 1.01) | +6.5 p.p. | dentro do limite |
| Drift de conceito | Escolaridade | ensino médio | 1318 | 0.82 (0.77 a 0.87) | +13.2 p.p. | inconclusivo |
| Drift de conceito | Estado civil | solteiro(a) | 4915 | 1.00 (0.96 a 1.04) | +1.5 p.p. | dentro do limite |

Maior diferença de oportunidade entre todos os grupos comparados: 13.2 p.p. (escolaridade: ensino médio, drift de conceito).

### Quanto o modelo ganharia usando os atributos protegidos

Mesmo algoritmo (xgboost), 3 seeds, com e sem sex, education, marriage, age como preditores.

| Variante | ROC AUC (validação) | ROC AUC (teste) |
|---|---|---|
| Sem atributos protegidos (modelo em produção) | 0.7754 | 0.7809 |
| Com atributos protegidos | 0.7782 | 0.7806 |
| **Ganho** | **+0.0028** | **-0.0003** |

## 6. Remediação (backtest em meses posteriores ao ajuste)

Ajuste com os rótulos maduros de 2025-01 a 2025-09 (atraso de 2 meses); teste em 2025-11, 2025-12. O pool de clientes é dividido ao meio: uma metade gera os meses de ajuste e a outra os de teste (1000 propostas por mês), e o experimento roda nas duas direções (4000 propostas de teste no total). Nenhum cliente aparece nos dois lados.

| Estratégia | Linhas de ajuste | ROC AUC (dobras) | KS | Aprovação no corte próprio | Inadimplência entre aprovados | Inadimplência entre aprovados, mesmo volume |
|---|---|---|---|---|---|---|
| Manter o modelo atual | 8989 | 0.610 (0.610 / 0.610) | 0.168 | 56.0% | 30.5% | 30.5% |
| Recalibrar a PD por segmento | 3000 | 0.618 (0.619 / 0.617) | 0.174 | 54.2% | 30.2% | 30.3% |
| Retreinar com a janela recente | 3000 | 0.625 (0.619 / 0.632) | 0.181 | 60.5% | 30.5% | 29.8% |
| Retreinar com referência + produção | 17989 | 0.631 (0.629 / 0.633) | 0.186 | 51.5% | 29.4% | 29.9% |
| Teto: retreino com rótulos imediatos (inviável) | 2000 | 0.646 (0.629 / 0.663) | 0.244 | 45.7% | 25.2% | 28.3% |

A última coluna aprova o mesmo volume em todas as estratégias (56.0%, a taxa de aprovação do modelo atual nos meses de teste), o que torna a inadimplência comparável.

Fatores da recalibração por segmento (inadimplência observada ÷ prevista nas safras recentes): em atraso 1.11, em dia, utilização alta 1.08, em dia, utilização baixa 0.88.

O teto é um modelo treinado nos próprios meses de teste, com outros clientes: mostra quanto um retreino recuperaria se o rótulo fosse imediato. A distância entre o teto e o desempenho da referência é a parte da perda que nenhum retreino recupera, porque o drift de conceito tornou a inadimplência menos previsível.

### Se o rótulo amadurecesse em 1 mês

Mesmo teste, com ajuste até 2025-10.

| Estratégia | ROC AUC (dobras) | Inadimplência entre aprovados, mesmo volume |
|---|---|---|
| Manter o modelo atual | 0.610 (0.610 / 0.610) | 30.5% |
| Recalibrar a PD por segmento | 0.638 (0.640 / 0.636) | 29.3% |
| Retreinar com a janela recente | 0.635 (0.638 / 0.632) | 29.2% |
| Retreinar com referência + produção | 0.638 (0.639 / 0.636) | 29.9% |
