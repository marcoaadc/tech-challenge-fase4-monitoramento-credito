# Contribuindo

## Setup

```bash
poetry install
poetry run pre-commit install
cp .env.example .env
make pipeline        # refaz dados, modelo, monitoramento, análises e dashboard
make test-fast       # testes de unidade, em segundos
```

## Fluxo de trabalho

1. Crie uma branch a partir de `main` (`feat/...`, `fix/...`, `docs/...`).
2. Rode `make lint test` antes de abrir o PR. O CI executa lint, testes em Python 3.10 a 3.12 e o pipeline completo em ambiente limpo.
3. Mensagens de commit seguem [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`, `test:`, `ci:`, `build:`, `chore:`, com escopo quando ajuda (`feat(contracts): ...`).
4. Registre mudanças relevantes em `CHANGELOG.md`, em uma seção *Não lançado* no topo (crie-a se não existir).

## Quando regenerar os artefatos

Os relatórios versionados em `reports/`, os dados em `data/` e o modelo em `models/` são a evidência do projeto e precisam refletir o código. Rode `make pipeline` e inclua os artefatos no mesmo PR quando mudar:

- uma regra do contrato de dados ou a limpeza;
- o cenário de simulação, um mecanismo de drift ou a seed;
- um limiar de drift ou de alerta;
- o modelo, as features ou o ponto de corte.

Se os números mudarem, atualize as tabelas do `README.md` e de `docs/`, e o Model Card se a mudança for no modelo.

## Convenções

- Python 3.10+, `ruff` (linhas de 120 colunas), *type hints* em funções públicas.
- Configuração só via `Settings` (`pydantic-settings`, prefixo `CREDITWATCH_`); nada de `os.environ` espalhado.
- Estágios do pipeline são funções `run_*` em `src/creditwatch/pipelines/`; a CLI apenas as chama.
- Toda regra nova do contrato entra no catálogo (`contracts/rules.py`) com identificador `DQ-xx` e ganha um injetor de defeito em `simulation/corruption.py`, que o teste parametrizado usa para provar que ela dispara sozinha.
- Toda função estatística nova tem teste com um caso de resposta conhecida.
- Experimento que treina em uns meses e testa em outros precisa respeitar a maturação do rótulo e separar clientes entre treino e teste (`remediation_experiment` valida os dois).
- Textos exibidos ao usuário (alertas, títulos de regra, relatórios) são acentuados; comentários e docstrings ficam em ASCII.

## Privacidade

- Nenhum dado pessoal em logs, eventos ou mensagens de erro: registre contagens, nomes de colunas e identificadores de regra.
- `data/quarantine/` e `data/predictions/` nunca entram no repositório.
- Exemplos e testes usam apenas dados fictícios. CPFs de exemplo vêm de `fake_cpf`, que os gera inválidos por construção.
- Uma coluna nova no contrato é uma decisão de tratamento de dados: justifique a finalidade no PR e atualize o inventário do README (seção 8.2).
