# ADR 002 — Tamanho de efeito (PSI) como critério de drift; p-valor apenas como evidência

**Status:** aceito · **Data:** 2026-11-05

## Contexto

Cada remessa de produção tem 2.000 propostas e é comparada com uma referência de cerca de 15.000. Com amostras desse tamanho, testes de hipótese (Kolmogorov-Smirnov, qui-quadrado) rejeitam a igualdade para diferenças sem nenhuma relevância prática: no último mês do cenário, **28 das 30 features têm p < 0,001**, inclusive as que mal se moveram. Um alerta baseado só em p-valor dispararia sempre e deixaria de informar.

## Decisão

1. A severidade de cada feature é decidida por **tamanho de efeito**:
   - PSI ≥ 0,25 → crítico; PSI ≥ 0,10 → atenção (faixas clássicas da indústria de crédito);
   - o teste de hipótese só eleva uma feature de estável para atenção quando o efeito também é relevante (D do KS nas numéricas ou distância de variação total nas categóricas ≥ 0,10 **e** p < 0,05).
2. Há *drift do dataset* quando 30% ou mais das features monitoradas estão em atenção ou crítico.
3. O detector é implementado no próprio projeto (`drift/statistics.py`, `drift/detector.py`) e o **Evidently** roda em paralelo com o mesmo método (PSI) e os mesmos limiares, gerando o relatório HTML e servindo de contraprova (`tests/test_integrations.py`).

## Alternativas consideradas

| Alternativa | Por que não |
|---|---|
| Só o Evidently, com o teste padrão por coluna | O padrão escolhe o teste pelo tamanho da amostra e decide por p-valor ou distância conforme o caso; os alertas ficariam dependentes de uma heurística que não controlamos. Além disso, um preset do Evidently aplica um único método por coluna, e o alerta precisa de PSI e do teste de hipótese lado a lado. |
| Só KS / qui-quadrado com α = 0,05 | Alarme permanente em amostras grandes (ver contexto). |
| Correção de múltiplos testes (Bonferroni/FDR) | Resolve o problema de testar 30 features, não o de amostras grandes: o p-valor continua minúsculo. |

## Consequências

- **Positivas:** nos três meses estáveis do cenário não há nenhum alarme falso; os limiares são configuráveis por variável de ambiente (`CREDITWATCH_PSI_*`) e documentados em [docs/observabilidade.md](../observabilidade.md); cada função estatística tem teste de unidade.
- **Negativas:** o PSI depende de como as faixas são montadas. O detector próprio usa decis da referência (faixas com bordas repetidas são fundidas, o que importa em variáveis com massa em zero) e o Evidently usa outro esquema; as duas contas concordam nas features críticas e podem divergir em casos de fronteira. Os dois valores ficam registrados no resumo do lote.
- **Limite conhecido:** drift nas entradas não é o mesmo que degradação. O drift de conceito não altera nenhuma feature e só aparece nas métricas com rótulo — por isso existem alertas separados de performance, calibração e taxa de inadimplência.
