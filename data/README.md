# Dados

## Fonte

**Default of Credit Card Clients** — Yeh, I-C. (2009). UCI Machine Learning Repository. <https://doi.org/10.24432/C55S3H>

- 30.000 clientes de cartão de crédito de um banco em Taiwan, com o histórico de abril a setembro de 2005 e o indicador de inadimplência no mês seguinte.
- Licença **Creative Commons Attribution 4.0 International (CC BY 4.0)**: uso, redistribuição e adaptação permitidos com atribuição.
- Artigo de referência: Yeh, I-C., & Lien, C-H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients*. Expert Systems with Applications, 36(2), 2473–2480.

O arquivo `raw/default_of_credit_card_clients.csv` é o conteúdo da planilha original convertido para CSV, **sem nenhuma alteração de valores**. Se ele não existir, `creditwatch.data.source` baixa o zip da UCI, confere o SHA-256 e refaz a conversão.

Os dados não trazem identificador direto (nome, documento, contato). Sexo, idade, estado civil e escolaridade são dados pessoais e recebem o tratamento descrito no README (governança e LGPD).

## Camadas

| Pasta | Conteúdo | Versionado |
|---|---|---|
| `raw/` | Snapshot bruto da UCI (30.000 linhas). **Reprova** no contrato de dados (códigos fora do dicionário e duplicatas). | sim |
| `reference/` | **Dataset de Referência**: metade do dataset limpo, com as partições `train`/`val`/`test`. É a única base usada para treinar e avaliar o modelo. | sim |
| `production/` | **Dataset de Produção**: doze remessas mensais simuladas (`batch_<mês>.parquet`, sem rótulo), os rótulos que amadurecem depois (`labels_<mês>.parquet`), a remessa corrompida da demonstração e o `manifest.json` com o cenário de cada mês. | sim |
| `interim/` | Metade do dataset limpo reservada para a simulação da produção; nunca é vista no treino. | não (recriado por `ingest`) |
| `predictions/` | Score e decisão de cada proposta escorada. | não |
| `quarantine/` | Remessas bloqueadas pelo contrato, com o relatório de validação. Pode conter dados pessoais: fora do repositório, acesso restrito, retenção curta. | não |

A remessa `batch_2025-07_remessa1_corrompida.parquet` contém uma coluna `cpf` com valores **fictícios e inválidos por construção** (o dígito verificador é deslocado de propósito), usada para demonstrar o bloqueio de PII pelo esquema estrito.

## Dicionário (nomes normalizados)

| Coluna | Original | Descrição |
|---|---|---|
| `customer_id` | `ID` | Identificador da proposta |
| `limit_bal` | `LIMIT_BAL` | Limite de crédito concedido (NT$) |
| `sex` | `SEX` | 1 masculino, 2 feminino |
| `education` | `EDUCATION` | 1 pós-graduação, 2 graduação, 3 ensino médio, 4 outros |
| `marriage` | `MARRIAGE` | 1 casado(a), 2 solteiro(a), 3 outros |
| `age` | `AGE` | Idade em anos |
| `pay_status_1..6` | `PAY_0`, `PAY_2..6` | Situação do pagamento, do mês mais recente (1) ao mais antigo (6): −2 sem consumo, −1 pago integralmente, 0 crédito rotativo, 1 a 8 meses de atraso |
| `bill_amt_1..6` | `BILL_AMT1..6` | Valor da fatura em cada mês (NT$) |
| `pay_amt_1..6` | `PAY_AMT1..6` | Valor pago em cada mês (NT$) |
| `default` | `default payment next month` | 1 se o cliente ficou inadimplente no mês seguinte |

O arquivo original nomeia o status mais recente como `PAY_0` e pula o `PAY_1`; a normalização corrige essa inconsistência.
