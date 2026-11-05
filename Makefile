# Tech Challenge FIAP Fase 04 - sustentacao e confiabilidade de um modelo de credit scoring
# Requer: Poetry >= 2.0 e make (no Windows, use Git Bash).

MLFLOW_URI ?= sqlite:///mlflow.db
BAD_BATCH  ?= data/production/batch_2025-07_remessa1_corrompida.parquet

.PHONY: help install lint format test test-fast test-cov pipeline ingest train simulate monitor analyze \
        dashboard demo-bad-batch evidently-all mlflow-ui serve-reports clean

help:
	@echo "Alvos disponiveis:"
	@echo "  install         Instala dependencias (poetry install)"
	@echo "  lint            ruff check + ruff format --check"
	@echo "  format          Formata o codigo e aplica fixes do ruff"
	@echo "  test            Roda a suite completa (inclui Evidently e MLflow reais, 1 a 3 min)"
	@echo "  test-fast       Roda apenas os testes de unidade (segundos)"
	@echo "  test-cov        Suite completa com relatorio de cobertura"
	@echo "  pipeline        ingest -> train -> simulate -> monitor -> analyze -> dashboard (com MLflow)"
	@echo "  demo-bad-batch  Valida o lote corrompido: o contrato bloqueia e o comando sai com codigo 1"
	@echo "  evidently-all   Regera o monitoramento com relatorio do Evidently para os 12 meses"
	@echo "  mlflow-ui       Abre a UI do MLflow em http://127.0.0.1:5000"
	@echo "  serve-reports   Serve reports/ em http://127.0.0.1:8000/dashboard/"
	@echo "  clean           Remove caches e artefatos temporarios"

install:
	poetry install

lint:
	poetry run ruff check .
	poetry run ruff format --check .

format:
	poetry run ruff format .
	poetry run ruff check --fix .

test:
	poetry run pytest

test-fast:
	poetry run pytest -m "not slow"

test-cov:
	poetry run pytest --cov --cov-report=term-missing

pipeline:
	CREDITWATCH_MLFLOW_TRACKING_URI=$(MLFLOW_URI) poetry run python -m creditwatch.pipelines.run_all

ingest:
	poetry run python -m creditwatch.pipelines.ingest

train:
	CREDITWATCH_MLFLOW_TRACKING_URI=$(MLFLOW_URI) poetry run python -m creditwatch.pipelines.train

simulate:
	poetry run python -m creditwatch.pipelines.simulate

monitor:
	CREDITWATCH_MLFLOW_TRACKING_URI=$(MLFLOW_URI) poetry run python -m creditwatch.pipelines.monitor

analyze:
	poetry run python -m creditwatch.pipelines.analyze

dashboard:
	poetry run python -m creditwatch.pipelines.dashboard

# O "-" faz o make seguir adiante: o codigo de saida 1 e o comportamento esperado (ingestao bloqueada).
demo-bad-batch:
	-poetry run python -m creditwatch.pipelines.validate $(BAD_BATCH)

evidently-all:
	poetry run python -m creditwatch.pipelines.monitor --evidently all
	poetry run python -m creditwatch.pipelines.dashboard

mlflow-ui:
	poetry run mlflow ui --backend-store-uri $(MLFLOW_URI) --host 127.0.0.1 --port 5000

serve-reports:
	poetry run python -m http.server 8000 --bind 127.0.0.1 --directory reports

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage coverage.xml ci-run logs data/interim data/predictions data/quarantine
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
