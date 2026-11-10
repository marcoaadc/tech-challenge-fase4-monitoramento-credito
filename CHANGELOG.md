# Changelog

Formato baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/); versionamento [SemVer](https://semver.org/lang/pt-BR/).

## [1.1.0] - 2026-11-05

Revisão completa da entrega. Os números do cenário mudaram (nova amostragem) e uma conclusão mudou: retreinar **não** recupera a queda causada pelo drift de conceito.

### Corrigido
- **Experimento de remediação com vazamento.** O mesmo cliente podia estar no treino e no teste, e o desafiante usava rótulos que ainda não existiriam na data da decisão. O backtest agora separa clientes em metades disjuntas, valida a maturação do rótulo e a ordem dos meses. Resultado honesto: ROC AUC de 0,610 para 0,631 na melhor estratégia (antes, 0,632 para 0,699).
- **Fração em coluna inteira passava no contrato.** A conversão de tipos truncava `17.9` para `17`; valores com fração agora reprovam em `DQ-10`.
- **Quarentena guardava colunas fora do contrato.** Um `cpf` enviado por engano era copiado para a quarentena; a cópia agora descarta essas colunas.
- **Métricas de referência do monitoramento** eram calculadas sobre treino, validação e teste juntos; passam a usar só a partição de teste.
- **Arquivo de rótulos inválido** derrubava a execução; agora o estágio é marcado como bloqueado, gera alerta e o monitoramento segue.
- `poetry.lock` fora de sincronia com o `pyproject.toml`.

### Adicionado
- Verificação de SHA-256 também do CSV extraído, além do arquivo baixado.
- Amostragem ponderada por Gumbel-top-k com números aleatórios comuns: os mecanismos de drift ficam comparáveis entre si, cliente a cliente.
- Atribuição causal com várias seeds e desvio-padrão dos valores de Shapley.
- Varredura de calibração por feature, que localiza onde o modelo erra sem usar conhecimento do cenário.
- Auditoria de viés com intervalo de confiança de 95%, veredito (dentro do limite, inconclusivo, impacto adverso) e diferença de oportunidade entre grupos.
- Backtest de remediação com teto de rótulos imediatos e cenário de maturação mais curta.
- Tempo de CPU por estágio (`cpu_ms`, `creditwatch_stage_cpu_seconds_total`), log da execução e resumo do MLflow versionados em `reports/observability/`.
- CI: `poetry check --lock`.

### Alterado
- Tamanho de efeito do qui-quadrado: distância de variação total no lugar do V de Cramér.
- Dashboard: card de varredura de calibração, intervalos na auditoria de viés e tratamento de meses sem rótulo.
- README e documentos reescritos com os novos números, glossário e mapa de requisitos do enunciado.

## [1.0.0] - 2026-11-05

### Adicionado
- Ingestão do dataset *Default of Credit Card Clients* (UCI) com verificação de SHA-256, limpeza documentada e separação entre Dataset de Referência e população de produção.
- Contrato de dados em Pandera com catálogo de 13 regras bloqueantes e 1 aviso, portão com quarentena e relatório por regra, e CLI de validação com código de saída.
- Injeção controlada de defeitos (12 tipos) para testar cada regra e montar o lote de demonstração.
- Modelo baseline de risco de crédito: Regressão Logística e XGBoost, seleção por ROC AUC, ponto de corte por KS, metadados com hashes e verificação de integridade na carga.
- Simulação de doze remessas mensais com quatro mecanismos isoláveis (inflação, aperto financeiro, novos perfis, drift de conceito) e rótulos entregues à parte, após a maturação.
- Detecção de drift por feature (PSI, Kolmogorov-Smirnov, qui-quadrado), drift de predição e drift do alvo, com relatório do Evidently nos checkpoints.
- Observabilidade: logs JSON centralizados com `run_id`, eventos de estágio, seis famílias de alerta, rastreamento no MLflow e exportação de métricas em formato Prometheus.
- Dashboard estático de monitoramento, em um único HTML sem dependências externas.
- Análises de governança: atribuição causal por valores de Shapley, diagnóstico por segmento, auditoria de viés, valor preditivo dos atributos protegidos e experimento de remediação.
- Documentação: README com plano de proteção de dados (LGPD) e análise causal, contrato de dados, drift, observabilidade, runbook, Model Card e quatro ADRs.
- CI (lint, testes em três versões do Python, pipeline completo em ambiente limpo), Dependabot e hooks de pre-commit.
