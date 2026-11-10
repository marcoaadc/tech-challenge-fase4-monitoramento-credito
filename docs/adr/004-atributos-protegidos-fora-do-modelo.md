# ADR 004 — Atributos sociodemográficos fora do modelo, mantidos para auditoria de viés

**Status:** aceito · **Data:** 2026-11-05

## Contexto

O dataset traz sexo, idade, estado civil e escolaridade de cada cliente. Usá-los como preditores de inadimplência faria a decisão de crédito depender diretamente de características pessoais que o titular não controla, em tensão com o princípio da não discriminação (LGPD, art. 6º, IX) e com a vedação de informações sensíveis e excessivas na análise de crédito (Lei 12.414/2011, art. 3º, § 3º).

O ganho de desempenho foi medido (`protected_attribute_ablation`, em `reports/analysis/fairness_audit.json`): treinando o mesmo XGBoost com e sem os quatro atributos, em três seeds, o ROC AUC de teste é **0,781 nos dois casos** (0,7806 com, 0,7809 sem); na validação, 0,7782 contra 0,7754. O histórico de pagamento já carrega a informação preditiva; os atributos pessoais só acrescentariam risco.

Por outro lado, apagar essas colunas impediria medir se o modelo trata os grupos de forma desigual: retirar um atributo das entradas não retira o seu efeito, porque as variáveis financeiras podem funcionar como *proxies*.

## Decisão

1. **O modelo usa apenas comportamento financeiro**: limite, status de pagamento, faturas, pagamentos e razões derivadas (`MODEL_FEATURES`, 26 colunas). Os quatro atributos sociodemográficos ficam em `PROTECTED` e um teste garante que as duas listas não se cruzam.
2. **Os atributos continuam no contrato de dados** por dois motivos: validar a capacidade civil (idade ≥ 18) e permitir a auditoria.
3. **A auditoria de viés roda a cada execução** (`analysis/fairness.py`): taxa de aprovação por grupo, razão de impacto desigual (regra dos quatro quintos) e taxa de aprovação entre bons pagadores, por período. Grupos com menos de 100 clientes ou de 2% da amostra ficam fora da comparação: a taxa de um grupo pequeno é ruidosa e, se ele virar o grupo de referência, fabrica impacto adverso nos demais.
4. Os atributos são **monitorados para drift de população**, mesmo fora do modelo: uma mudança de perfil da carteira é um sinal de negócio.

## Alternativas consideradas

| Alternativa | Por que não |
|---|---|
| Usar todos os atributos | Nenhum ganho mensurável de discriminação (ver contexto) e risco jurídico e reputacional alto; decisões passariam a variar com sexo e estado civil. |
| Remover os atributos na ingestão | Impede a auditoria e a validação de maioridade; "justiça por desconhecimento" não garante resultado equitativo. |
| Restrições de equidade no treino (reponderação, pós-processamento por grupo) | Exigiria usar o atributo protegido na decisão, o que a escolha acima evita; fica como opção se a auditoria mostrar impacto adverso persistente. |

## Consequências

- **Positivas:** a decisão é explicável só com variáveis financeiras (direito à revisão, LGPD art. 20); o metadado do modelo registra os atributos excluídos (`excluded_protected_attributes`).
- **Negativas:** a auditoria mostra que a exclusão **não elimina** a disparidade. O grupo de escolaridade "ensino médio" é aprovado a 0,87 da taxa do grupo mais aprovado na referência e a 0,83 durante o drift de dados, com intervalo de 95% que encosta em 0,80; entre bons pagadores, a diferença de aprovação chega a 13,2 p.p. O resultado está no README (mitigação de vieses) e o que fazer se o limite for cruzado, em [docs/runbook.md](../runbook.md).
- **Custo de privacidade:** manter os atributos significa tratá-los; por isso têm finalidade declarada (auditoria e validação), não entram em logs nem em relatórios de falha com identificação, e seguem o plano de retenção.
