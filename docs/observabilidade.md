# Observabilidade do pipeline e do modelo (Etapa 3)

O que é observado, onde fica registrado e como consultar. A decisão de arquitetura está no [ADR 003](adr/003-observabilidade-sem-servidor.md); o que fazer quando um alerta dispara, no [runbook](runbook.md).

## 1. Visão geral

```
 estágios do pipeline ──▶ StageTracker ──▶ reports/observability/pipeline_events.jsonl  (1 evento por estágio)
 (ingest, train,             │
  simulate, monitor,         ├─▶ logs/pipeline.jsonl          log estruturado, mesmo run_id em todos os estágios
  analyze, dashboard)        ├─▶ PipelineMetrics ──▶ metrics.prom (+ Pushgateway opcional)
                             └─▶ MLflow            ──▶ experimentos de treino e de monitoramento

 resumo de cada remessa ─▶ reports/monitoring/batches/<remessa>.json ─▶ timeline.csv · feature_drift.csv
                             └─▶ regras de alerta ─▶ reports/observability/alerts.jsonl

 tudo acima ─▶ reports/dashboard/index.html   (HTML único, abre offline)
```

Código em [`src/creditwatch/observability/`](../src/creditwatch/observability/) e [`src/creditwatch/dashboard/`](../src/creditwatch/dashboard/).

## 2. Logs centralizados

Todos os estágios escrevem no mesmo arquivo `logs/pipeline.jsonl` (JSON Lines) e no console. Cada linha traz o `run_id` da execução, então uma execução inteira se reconstrói com um filtro:

```json
{"timestamp": "2026-11-05T23:28:46.485+00:00", "level": "INFO", "logger": "creditwatch.pipeline",
 "message": "estagio contract [2025-01]: success em 130 ms", "event": "stage_finished",
 "pipeline": "monitor", "stage": "contract", "batch": "2025-01", "status": "success",
 "duration_ms": 130.39, "rows_in": 2000, "rows_out": 2000, "run_id": "20261105T232817-366de0b8"}
```

Os logs registram contagens, nomes de colunas e de regras — **nunca valores de linhas**. A pasta `logs/` não é versionada; os eventos e alertas consolidados ficam em `reports/observability/`.

## 3. Eventos de estágio (saúde operacional)

Cada estágio roda dentro de `tracker.stage(...)`, que mede a duração e classifica o resultado:

| Status | Significado |
|---|---|
| `success` | O estágio terminou normalmente |
| `blocked` | O estágio barrou a remessa de propósito (contrato reprovado); o pipeline segue para a próxima |
| `failed` | Erro inesperado; a exceção é propagada e a execução para |

Distinguir `blocked` de `failed` importa para o plantão: um bloqueio é o sistema funcionando; uma falha é um defeito do pipeline.

Última execução ([`pipeline_events.jsonl`](../reports/observability/pipeline_events.jsonl)), pipeline `monitor`:

| Estágio | Execuções | Sucesso | Bloqueio | Falha | Mediana | Máximo |
|---|---|---|---|---|---|---|
| `read_batch` | 13 | 13 | 0 | 0 | 12 ms | 15 ms |
| `contract` | 13 | 12 | 1 | 0 | 66 ms | 126 ms |
| `score` | 12 | 12 | 0 | 0 | 27 ms | 30 ms |
| `drift` | 12 | 12 | 0 | 0 | 133 ms | 153 ms |
| `performance` | 12 | 12 | 0 | 0 | 31 ms | 37 ms |
| `evidently` | 3 | 3 | 0 | 0 | 3,2 s | 6,7 s |

Treze remessas recebidas, doze escoradas (24.000 propostas) e uma bloqueada (2.040 linhas em quarentena). A escoragem leva cerca de 10 ms por mil propostas; os tempos variam com a máquina e com a carga do momento, e servem como ordem de grandeza.

## 4. Catálogo de métricas

### 4.1 Saúde do pipeline

| Métrica | Série Prometheus | O que responde | Atenção quando |
|---|---|---|---|
| Execuções de estágio por resultado | `creditwatch_stage_runs_total{stage,status}` | O pipeline está rodando? Algum estágio falha? | qualquer `status="failed"` |
| Duração dos estágios | `creditwatch_stage_duration_seconds{stage}` (histograma) | Algum estágio ficou lento? | p95 acima de 3× a mediana histórica |
| Remessas por resultado | `creditwatch_batches_total{status}` | Quantas remessas foram escoradas e quantas bloqueadas? | `blocked` > 0 |
| Volume processado | `creditwatch_rows_total{outcome}` (`scored`, `quarantined`) | O volume está dentro do esperado? | queda > 30% contra a média |
| Violações do contrato | `creditwatch_contract_violations_total{rule}` | Qual regra a origem está quebrando? | qualquer incremento |
| Última execução | `creditwatch_last_run_timestamp_seconds` | O job mensal rodou? | sem atualização há mais de 35 dias |
| Último lote | `creditwatch_last_batch_info{batch,model_version}` | Qual lote e qual modelo estão em produção? | — |

### 4.2 Drift (disponível no dia da escoragem)

| Métrica | Série Prometheus | Definição | Limiar |
|---|---|---|---|
| PSI por feature | `creditwatch_feature_psi{feature}` | PSI da feature contra a referência | atenção ≥ 0,10 · crítico ≥ 0,25 |
| Features com drift | `creditwatch_drifted_features` | Número de features em atenção ou crítico | — |
| Fração com drift | `creditwatch_drift_share` | Features com drift ÷ features monitoradas | drift do dataset ≥ 0,30 |
| PSI do score | `creditwatch_score_psi` | PSI do score contra o score da referência fora do treino | atenção ≥ 0,10 · crítico ≥ 0,25 |
| Taxa de aprovação | `creditwatch_approval_rate` | Fração de propostas com PD abaixo do ponto de corte | sem limiar fixo; métrica de negócio |

### 4.3 Saúde do modelo (disponível após a maturação do rótulo)

| Métrica | Série Prometheus | Definição | Limiar |
|---|---|---|---|
| ROC AUC | `creditwatch_model_roc_auc` | Poder de ordenação entre bons e maus pagadores | queda ≥ 0,03 atenção · ≥ 0,05 crítico |
| KS | `creditwatch_model_ks` | Maior separação entre as distribuições de score | acompanha o AUC |
| Taxa de inadimplência | `creditwatch_default_rate` | Fração de inadimplentes no lote | mudança ≥ 5 p.p. com p < 0,05 |
| Calibração | — (`calibration_gap` no resumo do lote) | Inadimplência observada − PD média prevista | \|diferença\| ≥ 5 p.p. |
| Inadimplência entre aprovados | `creditwatch_bad_rate_approved` | Inadimplentes ÷ aprovados: a perda que chega à carteira | métrica de negócio; referência 12,3% |

Todos os limiares são configuráveis por variável de ambiente (`.env.example`).

### 4.4 Alertas

| Família (`kind`) | Dispara quando | Depende de rótulo |
|---|---|---|
| `data_quality` | Remessa bloqueada pelo contrato | não |
| `data_drift` | Feature em atenção ou crítico, ou drift do dataset — **no máximo um alerta por remessa**, listando as features mais afetadas | não |
| `prediction_drift` | PSI do score ≥ 0,10 | não |
| `performance` | Queda de ROC AUC ≥ 0,03 | sim |
| `calibration` | Inadimplência observada distante da PD prevista | sim |
| `target_drift` | Taxa de inadimplência mudou de forma significativa e relevante | sim |

Cada alerta de [`alerts.jsonl`](../reports/observability/alerts.jsonl) traz severidade, métrica, valor, limiar e `detectable_at`: o mês em que poderia disparar em produção. Para as famílias que dependem de rótulo, é o mês da remessa mais o atraso de maturação — a queda de performance de setembro só é detectável em novembro.

Na execução de referência foram **29 alertas (18 críticos e 11 de atenção)**: 9 de drift de dados, 7 de drift do alvo, 6 de drift de predição, 4 de performance, 2 de calibração e 1 de qualidade de dados. A série `creditwatch_alerts_total{kind,severity}` traz a contagem.

## 5. MLflow

```bash
make pipeline      # roda tudo com CREDITWATCH_MLFLOW_TRACKING_URI=sqlite:///mlflow.db
make mlflow-ui     # http://127.0.0.1:5000
```

| Experimento | Runs | Conteúdo |
|---|---|---|
| `creditwatch-treino` | 1 run pai + 1 filho por candidato | Hiperparâmetros, métricas de validação de cada candidato, métricas de teste do escolhido, `metadata.json` e o modelo com assinatura, registrado como `credit-scoring-baseline` no Model Registry |
| `creditwatch-monitoramento` | 1 run pai por execução + 1 filho por remessa | No run pai, as métricas de cada remessa são logadas com `step`, e a aba de métricas mostra a série temporal de `performance/roc_auc`, `drift/n_drifted`, `psi/<feature>`, `prediction/score_psi` e `alerts/critical`. Cada run filho guarda o plano do mês como parâmetros, o resumo JSON e o relatório do Evidently como artefatos |

O modelo é registrado no formato nativo de cada biblioteca, sem pickle: XGBoost em UBJ (`mlflow.xgboost`) e pipelines scikit-learn em skops (`mlflow.sklearn`).

Sem a variável `CREDITWATCH_MLFLOW_TRACKING_URI` o rastreamento fica desligado e o pipeline roda igual: uma falha de observabilidade não pode impedir uma decisão de crédito.

## 6. Prometheus (integração opcional com a Fase 3)

O monitoramento grava [`metrics.prom`](../reports/observability/metrics.prom) no formato de exposição do Prometheus:

```
creditwatch_batches_total{status="scored"} 12.0
creditwatch_batches_total{status="blocked"} 1.0
creditwatch_contract_violations_total{rule="DQ-09"} 1.0
creditwatch_feature_psi{feature="utilization_1"} 0.90483
creditwatch_model_roc_auc 0.6129206151967529
creditwatch_alerts_total{kind="data_drift",severity="critical"} 8.0
```

Duas formas de levar essas séries a um Prometheus existente:

- **Textfile collector** do node_exporter: aponte `--collector.textfile.directory` para a pasta do arquivo.
- **Pushgateway**: defina `CREDITWATCH_PUSHGATEWAY_URL` e o job envia as métricas ao terminar (o envio é opcional e uma falha de rede só gera um aviso no log).

Regras de alerta equivalentes às do projeto, em PromQL:

```yaml
groups:
  - name: creditwatch
    rules:
      - alert: LoteBloqueadoPeloContrato
        expr: increase(creditwatch_batches_total{status="blocked"}[1d]) > 0
        labels: { severity: critical }
      - alert: DriftDoDataset
        expr: creditwatch_drift_share >= 0.30
        labels: { severity: critical }
      - alert: ModeloDegradado
        expr: creditwatch_model_roc_auc < 0.727   # referência 0,777 menos 0,05
        labels: { severity: critical }
      - alert: MonitoramentoParado
        expr: time() - creditwatch_last_run_timestamp_seconds > 35 * 86400
        labels: { severity: warning }
```

Essas regras são um exemplo de integração e não fazem parte do que o repositório executa.

## 7. Dashboard

[`reports/dashboard/index.html`](../reports/dashboard/index.html) é gerado por `python -m creditwatch.pipelines.dashboard` a partir dos artefatos acima. É um HTML único, sem script externo, com tema claro e escuro.

| Seção | O que mostra |
|---|---|
| Indicadores | ROC AUC e inadimplência entre aprovados no último mês contra a referência, features com drift, alertas, remessas e volume |
| Saúde do modelo | ROC AUC por mês com o limite de alerta; inadimplência observada × PD prevista × inadimplência entre aprovados |
| Drift dos dados de entrada | Features com drift por mês (atenção e crítico), PSI das três features mais afetadas, mapa de PSI por feature e mês |
| Detalhe do mês | Seletor de mês: PSI, teste, estatística, p-valor e médias de cada feature; alertas do mês; link para o relatório do Evidently |
| Saúde do pipeline | Situação de cada remessa, execuções e duração por estágio, regras violadas na remessa bloqueada |
| Causa da degradação | Atribuição da queda de AUC por mecanismo, inadimplência observada × prevista por segmento, remediação, auditoria de viés |
| Todos os alertas | Lista completa com severidade, tipo e mês em que cada alerta é detectável |

Todo gráfico tem uma tabela equivalente ("Ver tabela") e responde ao teclado (setas esquerda e direita). Para abrir localmente: `make serve-reports` e acessar <http://127.0.0.1:8000/dashboard/>, ou abrir o arquivo direto no navegador.
