# Changelog

Formato baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/); versionamento [SemVer](https://semver.org/lang/pt-BR/).

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
