# Runbook: o que fazer quando um alerta dispara

Guia de resposta para quem opera o modelo de crédito. Cada alerta de [`alerts.jsonl`](../reports/observability/alerts.jsonl) tem uma família (`kind`) e uma severidade; a tabela abaixo diz quem age, em quanto tempo e por onde começar.

| Alerta | Severidade típica | Responsável | Prazo de resposta |
|---|---|---|---|
| `data_quality` — remessa bloqueada | crítico | Engenharia de dados + dono do sistema de origem | no mesmo dia |
| `data_drift` — atenção | atenção | Ciência de dados | próxima revisão mensal |
| `data_drift` — drift do dataset | crítico | Ciência de dados + risco de crédito | 5 dias úteis |
| `prediction_drift` | atenção/crítico | Risco de crédito | 5 dias úteis |
| `performance` | crítico | Comitê de modelos | 2 dias úteis |
| `calibration` / `target_drift` | atenção/crítico | Risco de crédito | 5 dias úteis |

## 1. Remessa bloqueada pelo contrato (`data_quality`)

**O que significa:** a remessa violou uma regra rígida e não foi escorada. Nenhuma decisão de crédito foi tomada com dado inválido.

1. Abrir `reports/data_quality/blocked_<remessa>.json`: quais regras, quantas linhas, quais colunas.
2. Classificar a causa pela regra:
   - `DQ-09` (coluna extra ou ausente) → mudança de layout na origem. Se a coluna extra for um dado pessoal (ex.: `cpf`), tratar como **incidente de privacidade**: avisar o encarregado e apagar a cópia em quarentena assim que a origem confirmar a correção.
   - `DQ-01`, `DQ-10`, `DQ-11` → falha de extração ou de transferência; pedir o reenvio.
   - `DQ-02`, `DQ-03`, `DQ-05` a `DQ-08` → erro de cadastro ou de unidade; a origem corrige.
   - `DQ-04` → reenvio parcial; deduplicar na origem.
3. Não editar o lote à mão nem relaxar a regra para "fazer passar". A correção é na origem; a remessa corrigida entra como nova entrega (`<mês>-r2`).
4. Se a mesma regra quebrar em duas remessas seguidas, abrir uma ação corretiva com o dono do sistema de origem.

**Não é o caso de:** retreinar o modelo ou mexer em limiares.

## 2. Drift de dados (`data_drift`)

**O que significa:** a distribuição das entradas mudou em relação ao treino. O modelo pode ou não ter piorado — o drift de dados sozinho não responde isso.

1. Abrir o dashboard no mês do alerta e ordenar as features por PSI. Abrir também o relatório do Evidently do checkpoint mais próximo.
2. Perguntar **por que** mudou, nesta ordem:
   - **Erro de dado disfarçado?** Uma mudança brusca em uma única feature, de um mês para o outro, costuma ser alteração de unidade ou de regra de cálculo na origem. Se for, vira regra nova no contrato.
   - **Mudança de população?** Idade e limite se movendo juntos indicam novo canal de aquisição ou campanha. Confirmar com a área comercial.
   - **Mudança econômica?** Utilização, razão de pagamento e atrasos subindo juntos indicam aperto financeiro da base.
3. Verificar o que o drift faz com o negócio **hoje**, sem esperar o rótulo: taxa de aprovação e PSI do score. Uma queda forte de aprovação é decisão de política, não de modelo.
4. Priorizar pelas features que pesam no modelo: a tabela "PSI × importância" de [`analysis_report.md`](../reports/analysis/analysis_report.md) separa o drift que afeta o score do drift que só faz barulho.
5. Marcar a revisão de performance para quando o rótulo dessas safras amadurecer (2 meses).

**Não retreinar por reflexo.** No cenário deste projeto, 23 features entram em drift entre abril e agosto e o ROC AUC não cai; retreinar ali gastaria esforço e trocaria um modelo validado por outro sem ganho.

## 3. Drift de predição (`prediction_drift`)

**O que significa:** a distribuição do score mudou; a carteira aprovada tem outro perfil de risco.

1. Comparar a taxa de aprovação do mês com a da referência.
2. Decidir com a área de risco se o ponto de corte continua adequado ao apetite de risco. Mudar o corte é uma decisão de política e deve ficar registrada, com data e responsável.
3. Acompanhar a inadimplência entre os aprovados quando o rótulo amadurecer.

## 4. Queda de performance (`performance`)

**O que significa:** com o rótulo maduro, o modelo ordena pior do que na referência. É o alerta que indica perda financeira.

1. **Conter.** Enquanto a causa é investigada:
   - endurecer o ponto de corte ou reduzir limites nos segmentos afetados;
   - encaminhar para análise humana as propostas na faixa de score próxima do corte.
2. **Diagnosticar.** Rodar `make analyze` e ler `reports/analysis/`:
   - *Inadimplência observada × prevista por segmento*: um segmento em que a observada descola da prevista, sem drift nas suas features, é drift de conceito localizado.
   - Se o drift de dados é forte e a calibração por segmento está preservada, o problema é de população, e recalibrar pode bastar.
3. **Corrigir.** Avaliar um desafiante treinado com as safras recentes já maduras. O experimento de remediação compara as estratégias em meses posteriores ao treino; nos dados deste projeto, retreinar com a janela recente leva o ROC AUC de 0,632 para 0,699 e a inadimplência entre aprovados de 29,7% para 26,1% — recupera parte, não tudo, porque a mudança continua se intensificando.
4. **Promover com governança.** O desafiante só substitui o campeão depois de: passar pelos mesmos testes do contrato e de drift, repetir a auditoria de viés, ter o model card atualizado e a aprovação do comitê registrada. A versão anterior fica disponível para reversão.
5. **Reduzir a janela cega.** A queda de setembro só é detectável em novembro. Se o atraso for inaceitável, monitorar indicadores antecipados (atraso de 30 dias em vez de inadimplência plena) como rótulo provisório.

## 5. Calibração e taxa de inadimplência (`calibration`, `target_drift`)

**O que significa:** a inadimplência observada se afastou da prevista, ou a taxa de inadimplência da carteira mudou.

1. Se o AUC está preservado e só a calibração se moveu, recalibrar as probabilidades (o ordenamento continua válido) e rever provisões.
2. Se o AUC também caiu, seguir a seção 4.
3. Uma taxa de inadimplência maior com modelo calibrado é informação de risco de carteira, não defeito do modelo: comunicar à área de provisão.

## 6. Impacto adverso na auditoria de viés

**O que significa:** a taxa de aprovação de um grupo ficou abaixo de 80% da taxa do grupo mais aprovado.

1. Verificar se a diferença tem explicação de risco: comparar a inadimplência observada e a taxa de aprovação **entre bons pagadores** do grupo (diferença de oportunidade). Se bons pagadores do grupo são negados com mais frequência, o modelo está errando mais com eles.
2. Investigar *proxies*: quais features financeiras separam o grupo dos demais.
3. Registrar a análise e a decisão (manter, ajustar corte, rever features, revisão humana das negativas do grupo) e levar ao encarregado e ao comitê de modelos.
4. Garantir o canal de revisão da decisão automatizada para os titulares afetados (LGPD, art. 20).

No cenário deste projeto nenhum grupo cruza o limite, mas a distância diminui com o drift: a razão do grupo de escolaridade "ensino médio" vai de 0,87 na referência para 0,83 durante o drift de dados, quando a parcela desse grupo com atraso recente sobe de 27% (meses estáveis) para 42% e o modelo, que reage a atrasos, passa a negá-lo mais.

Antes de concluir que há impacto adverso, conferir o tamanho dos grupos: a auditoria só compara grupos com pelo menos 100 clientes e 2% da amostra. Sem esse piso, um grupo de 122 clientes com 82% de aprovação virava a referência e fazia "ensino médio" aparecer com 0,72.

## 7. Falha de estágio (`status = failed` nos eventos)

1. Localizar o evento em `pipeline_events.jsonl` e o *traceback* em `logs/pipeline.jsonl`, filtrando pelo `run_id`.
2. O pipeline é idempotente por remessa: corrigida a causa, basta rodar o estágio de novo.
3. Se o artefato do modelo falhar na verificação de integridade (SHA-256 diferente do registrado em `metadata.json`), **não** usar `verify_integrity=False` em produção: restaurar o artefato do registro de modelos e investigar a alteração.
