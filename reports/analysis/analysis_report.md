# Análises de governança: causalidade, viés e remediação

Gerado por `python -m creditwatch.pipelines.analyze`. Os números mudam se o cenário ou a seed mudarem.

## 1. O que causou a degradação (atribuição por mecanismo)

Intervenções sobre o plano de 2025-12 (6000 propostas por combinação). Cada célula é a parcela do efeito total atribuída ao mecanismo (valores de Shapley).

| Mecanismo | ROC AUC | KS | Inadimplência | Inadimplência entre aprovados | Features com drift |
|---|---|---|---|---|---|
| Inflação (faturas nominais) | -0.012 | -0.018 | +0.012 | +0.019 | +7.7 |
| Aperto financeiro (mais atrasos) | +0.017 | +0.034 | +0.028 | -0.008 | +7.2 |
| Novos perfis (jovens, limite baixo) | -0.002 | -0.005 | +0.019 | +0.011 | +3.2 |
| Drift de conceito (P(y|x)) | -0.161 | -0.279 | +0.066 | +0.145 | +0.0 |
| **Efeito total** | **-0.157** | **-0.269** | **+0.124** | **+0.167** | **+18.0** |

## 2. Onde a relação mudou (inadimplência observada x prevista por segmento)

| Período | Segmento | Participação | Observada | Prevista | Diferença |
|---|---|---|---|---|---|
| Referência | em dia, baixa utilização | 45.4% | 13.5% | 13.8% | -0.3% |
| Referência | em dia, alavancado | 31.8% | 14.4% | 14.0% | +0.4% |
| Referência | em atraso | 22.8% | 50.0% | 51.1% | -1.0% |
| Estável | em dia, baixa utilização | 45.8% | 11.2% | 13.2% | -2.0% |
| Estável | em dia, alavancado | 31.4% | 15.8% | 14.0% | +1.7% |
| Estável | em atraso | 22.8% | 49.7% | 49.4% | +0.3% |
| Drift de dados | em dia, baixa utilização | 34.0% | 13.5% | 13.8% | -0.3% |
| Drift de dados | em dia, alavancado | 30.4% | 15.3% | 15.3% | +0.1% |
| Drift de dados | em atraso | 35.7% | 56.6% | 51.0% | +5.5% |
| Drift de conceito | em dia, baixa utilização | 24.3% | 14.8% | 15.4% | -0.5% |
| Drift de conceito | em dia, alavancado | 29.2% | 37.1% | 15.7% | +21.4% |
| Drift de conceito | em atraso | 46.5% | 47.0% | 50.0% | -3.0% |

## 3. Features candidatas a causa do drift de predição (PSI x importância)

| Feature | PSI | Importância | Impacto |
|---|---|---|---|
| `max_delay` | 0.279 | 0.265 | 0.0739 |
| `pay_status_1` | 0.302 | 0.190 | 0.0573 |
| `months_delayed` | 0.291 | 0.167 | 0.0485 |
| `utilization_1` | 0.905 | 0.018 | 0.0160 |
| `utilization_avg` | 0.570 | 0.019 | 0.0108 |
| `pay_status_2` | 0.344 | 0.031 | 0.0106 |
| `bill_trend` | 0.658 | 0.014 | 0.0095 |
| `payment_ratio_1` | 0.519 | 0.016 | 0.0081 |
| `payment_ratio_avg` | 0.387 | 0.014 | 0.0054 |
| `pay_status_3` | 0.208 | 0.023 | 0.0047 |

## 4. Auditoria de viés (grupo menos aprovado de cada atributo)

| Período | Atributo | Grupo | Impacto desigual | Diferença de oportunidade | Impacto adverso (< 0,80) |
|---|---|---|---|---|---|
| Referência | Sexo | masculino | 0.94 | +0.018 | não |
| Referência | Faixa etária | 18-29 | 0.95 | +0.028 | não |
| Referência | Escolaridade | ensino médio | 0.87 | +0.090 | não |
| Referência | Estado civil | casado(a) | 0.98 | +0.011 | não |
| Estável | Sexo | masculino | 0.96 | +0.029 | não |
| Estável | Faixa etária | 18-29 | 0.93 | +0.046 | não |
| Estável | Escolaridade | ensino médio | 0.86 | +0.076 | não |
| Estável | Estado civil | casado(a) | 0.99 | +0.000 | não |
| Drift de dados | Sexo | masculino | 0.94 | +0.031 | não |
| Drift de dados | Faixa etária | 18-29 | 0.91 | +0.049 | não |
| Drift de dados | Escolaridade | ensino médio | 0.83 | +0.083 | não |
| Drift de dados | Estado civil | casado(a) | 1.00 | +0.000 | não |
| Drift de conceito | Sexo | masculino | 0.91 | +0.058 | não |
| Drift de conceito | Faixa etária | 50+ | 0.84 | +0.070 | não |
| Drift de conceito | Escolaridade | graduação | 0.84 | +0.107 | não |
| Drift de conceito | Estado civil | casado(a) | 0.96 | +0.000 | não |

### Quanto o modelo ganharia usando os atributos protegidos

Mesmo algoritmo (xgboost), 3 seeds, com e sem sex, education, marriage, age como preditores.

| Variante | ROC AUC (validação) | ROC AUC (teste) |
|---|---|---|
| Sem atributos protegidos (modelo em produção) | 0.7754 | 0.7809 |
| Com atributos protegidos | 0.7782 | 0.7806 |
| **Ganho** | **+0.0028** | **-0.0003** |

## 5. Remediação (avaliada em meses posteriores ao treino)

Treino com rótulos maduros de 2025-01 a 2025-10; teste em 2025-11, 2025-12 (4000 propostas).

| Estratégia | Linhas de treino | ROC AUC | KS | Aprovação | Inadimplência entre aprovados |
|---|---|---|---|---|---|
| `campeao` | 8989 | 0.632 | 0.186 | 55.4% | 29.7% |
| `retreino_janela_recente` | 6000 | 0.699 | 0.283 | 58.3% | 26.1% |
| `retreino_referencia_mais_producao` | 28989 | 0.686 | 0.261 | 49.5% | 26.4% |
