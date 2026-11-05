# ADR 003 — Observabilidade de um job em lote: eventos estruturados, MLflow e dashboard estático

**Status:** aceito · **Data:** 2026-11-05

## Contexto

A Etapa 3 pede logs e métricas centralizados, rastreamento (MLflow ou dashboard interativo) e métricas de saúde documentadas. O sistema monitorado aqui não é uma API de longa duração, como na Fase 3: é um **pipeline em lote** que roda uma vez por remessa mensal e termina. Não há processo vivo para um Prometheus raspar, e o avaliador precisa ver o resultado sem subir infraestrutura.

## Decisão

Uma única fonte de verdade — os **eventos de estágio** (`reports/observability/pipeline_events.jsonl`) e os **resumos por remessa** (`reports/monitoring/batches/*.json`) — alimenta três saídas:

| Saída | Para quem | Como |
|---|---|---|
| **Dashboard estático** (`reports/dashboard/index.html`) | Quem avalia ou opera, sem instalar nada | Um HTML único, com os dados embutidos e gráficos SVG desenhados no navegador; abre offline. |
| **MLflow** (`sqlite:///mlflow.db`) | Time de ML | Experimento de treino (candidatos, modelo registrado) e experimento de monitoramento (um run por execução com as métricas de cada remessa logadas com `step`, e um run filho por remessa com os relatórios). |
| **Prometheus** (`metrics.prom` + Pushgateway opcional) | Stack de monitoramento existente (Fase 3) | Arquivo de exposição para o *textfile collector* do node_exporter; envio a um Pushgateway se `CREDITWATCH_PUSHGATEWAY_URL` estiver definida. |

Os logs de todos os estágios vão para um único arquivo JSON Lines (`logs/pipeline.jsonl`) com o `run_id` da execução.

## Alternativas consideradas

| Alternativa | Por que não |
|---|---|
| Prometheus + Grafana em Docker Compose (como na Fase 3) | Exige um endpoint vivo ou um Pushgateway e infraestrutura que o leitor teria de subir; para um job mensal, as séries teriam um ponto por mês. As métricas continuam exportadas no formato Prometheus, então a integração é possível sem mudar código. |
| UI do Evidently (workspace + `evidently ui`) | Boa para explorar snapshots, mas é mais um servidor para subir e não cobre saúde do pipeline nem alertas de contrato. |
| Dashboard em Streamlit/Dash | Precisa de um processo Python rodando; um HTML estático pode ser versionado, anexado a um ticket e publicado como página. |
| Datadog / serviço gerenciado | Custo e credenciais para um projeto acadêmico; o formato dos eventos (JSON) e das métricas (Prometheus) é o que esses serviços ingerem. |

## Consequências

- **Positivas:** tudo o que é exibido pode ser reproduzido com `make pipeline` e conferido nos arquivos de origem; o dashboard não tem dependência externa (nenhum script de CDN); falha de observabilidade (MLflow desligado, Pushgateway fora do ar) não derruba o pipeline.
- **Negativas:** o dashboard é uma fotografia da última execução, sem atualização em tempo real; o arquivo Prometheus reflete o estado do último lote, e o histórico fica no MLflow e nos CSVs.
- **Evolução natural:** com um orquestrador (Airflow, como na Fase 3) chamando `creditwatch.pipelines.monitor` por remessa, o Pushgateway passa a receber uma amostra por execução e os alertas podem ser roteados pelo Alertmanager.
