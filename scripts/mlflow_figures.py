"""Gera figuras estaticas a partir do banco do MLflow, para a documentacao.

O banco (``mlflow.db``) nao e versionado; as figuras em ``docs/img/`` sao a evidencia do que o
rastreamento guarda. Uso, na raiz do projeto, depois de ``make pipeline``:

    poetry run python scripts/mlflow_figures.py [--tracking-uri sqlite:///mlflow.db] [--out docs/img]

Depende de matplotlib, instalado junto com o mlflow.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from matplotlib.ticker import MaxNLocator

TRAINING_EXPERIMENT = "creditwatch-treino"
MONITORING_EXPERIMENT = "creditwatch-monitoramento"

# Mesma paleta do dashboard (ver src/creditwatch/dashboard/template.html.j2).
INK = "#262625"
INK_SOFT = "#898781"
GRID = "#e8e7e3"
BLUE = "#3987e5"
BLUE_DARK = "#256abf"
ORANGE = "#d95926"
GRAY = "#c3c2b7"
CRITICAL = "#d03b3b"
WARNING = "#fab219"

MONTHS = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
PHASES = {"estavel": "Estável", "drift_de_dados": "Drift de dados", "drift_de_conceito": "Drift de conceito"}


def br(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def pct(value: float, _pos=None) -> str:
    return f"{value * 100:.0f}%"


def delivery_label(run_name: str) -> str:
    month, _, suffix = run_name.partition("-r")
    label = MONTHS[int(month[5:7]) - 1]
    return f"{label} r{suffix}" if suffix else label


def style_axes(ax) -> None:
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=8.5, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.title.set_color(INK)


def phase_markers(ax, phases: list[str], top: float) -> None:
    """Linhas verticais onde a fase muda, com o nome da fase a direita da linha."""
    for index, phase in enumerate(phases):
        if index == 0 or phase != phases[index - 1]:
            if index:
                ax.axvline(index - 0.5, color=GRAY, linewidth=1, linestyle=(0, (4, 3)))
            ax.text(index - 0.4, top, PHASES.get(phase, phase), color=INK_SOFT, fontsize=8, va="top")


def monitoring_figure(client: mlflow.MlflowClient, out: Path) -> Path:
    experiment = client.get_experiment_by_name(MONITORING_EXPERIMENT)
    if experiment is None:
        raise SystemExit(f"experimento {MONITORING_EXPERIMENT!r} nao encontrado; rode o pipeline com MLflow ligado")
    runs = client.search_runs([experiment.experiment_id], order_by=["attributes.start_time DESC"])
    parents = [r for r in runs if "mlflow.parentRunId" not in r.data.tags]
    parent = parents[0]  # execucao mais recente
    children = sorted(
        (r for r in runs if r.data.tags.get("mlflow.parentRunId") == parent.info.run_id),
        key=lambda r: r.info.start_time,
    )
    labels = [delivery_label(r.info.run_name) for r in children]
    phases = [r.data.params.get("phase", "") for r in children]
    steps = list(range(len(children)))

    def series(metric: str) -> list[float | None]:
        values: dict[int, float] = {m.step: m.value for m in client.get_metric_history(parent.info.run_id, metric)}
        return [values.get(step) for step in steps]

    roc = series("performance/roc_auc")
    reference = parent.data.metrics["reference/roc_auc"]
    blocked = [step for step, value in enumerate(series("pipeline/blocked")) if value]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), dpi=130)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"MLflow · experimento {MONITORING_EXPERIMENT} · run {parent.info.run_name}"
        " · métricas do run pai por remessa (step)",
        color=INK,
        fontsize=10.5,
        x=0.03,
        ha="left",
        y=0.985,
    )

    ax = axes[0][0]
    style_axes(ax)
    ax.set_title("performance/roc_auc", loc="left", fontsize=10)
    ax.plot(
        steps, roc, color=BLUE, linewidth=2, marker="o", markersize=4.5, markerfacecolor="white", markeredgewidth=1.6
    )
    ax.axhline(reference, color=INK_SOFT, linewidth=1, linestyle=(0, (4, 3)))
    ax.axhline(reference - 0.05, color=CRITICAL, linewidth=1, linestyle=(0, (1, 2)))
    ax.text(len(steps) - 0.6, reference + 0.004, f"referência {br(reference)}", color=INK_SOFT, fontsize=8, ha="right")
    ax.text(
        len(steps) - 0.6,
        reference - 0.05 - 0.013,
        "limite crítico (queda de 0,05)",
        color=CRITICAL,
        fontsize=8,
        ha="right",
    )
    last = max(step for step, value in enumerate(roc) if value is not None)
    ax.annotate(
        br(roc[last]),
        (last, roc[last]),
        textcoords="offset points",
        xytext=(0, -14),
        ha="center",
        color=INK,
        fontsize=8.5,
    )
    for step in blocked:
        ax.plot([step], [reference - 0.11], marker="x", color=CRITICAL, markersize=7, markeredgewidth=1.6)
        ax.text(
            step, reference - 0.135, "bloqueada\npelo contrato", color=CRITICAL, fontsize=7.5, ha="center", va="top"
        )
    ax.set_ylim(0.55, 0.86)
    phase_markers(ax, phases, 0.855)
    ax.set_xticks(steps, labels, rotation=0)

    ax = axes[0][1]
    style_axes(ax)
    ax.set_title("drift/n_drifted  (features com PSI ≥ 0,10, de 30)", loc="left", fontsize=10)
    drifted = [value or 0 for value in series("drift/n_drifted")]
    bars = ax.bar(steps, drifted, color=BLUE, width=0.62)
    for bar, value in zip(bars, drifted, strict=False):
        if value:
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.6, f"{value:.0f}", ha="center", color=INK, fontsize=8)
    ax.set_ylim(0, 30)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    phase_markers(ax, phases, 29.5)
    ax.set_xticks(steps, labels)

    ax = axes[1][0]
    style_axes(ax)
    ax.set_title("prediction/score_psi  (PSI da distribuição dos scores)", loc="left", fontsize=10)
    psi = series("prediction/score_psi")
    ax.plot(
        steps, psi, color=ORANGE, linewidth=2, marker="o", markersize=4.5, markerfacecolor="white", markeredgewidth=1.6
    )
    ax.axhline(0.10, color=WARNING, linewidth=1, linestyle=(0, (1, 2)))
    ax.axhline(0.25, color=CRITICAL, linewidth=1, linestyle=(0, (1, 2)))
    ax.text(-0.4, 0.107, "atenção 0,10", color="#8a5d00", fontsize=8)
    ax.text(-0.4, 0.257, "crítico 0,25", color=CRITICAL, fontsize=8)
    ax.set_ylim(0, 0.34)
    phase_markers(ax, phases, 0.335)
    ax.set_xticks(steps, labels)

    ax = axes[1][1]
    style_axes(ax)
    ax.set_title("alerts/critical e alerts/warning  (alertas emitidos na remessa)", loc="left", fontsize=10)
    critical = [value or 0 for value in series("alerts/critical")]
    warning = [value or 0 for value in series("alerts/warning")]
    ax.bar(steps, critical, color=CRITICAL, width=0.62, label="crítico")
    ax.bar(steps, warning, bottom=critical, color=WARNING, width=0.62, label="atenção", edgecolor="white", linewidth=1)
    for step in blocked:
        ax.text(step, critical[step] + warning[step] + 0.15, "contrato", color=CRITICAL, fontsize=7.5, ha="center")
    ax.set_ylim(0, max(c + w for c, w in zip(critical, warning, strict=False)) + 1.2)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK)
    ax.set_xticks(steps, labels)

    fig.text(
        0.03,
        0.012,
        "Fonte: get_metric_history do run pai; steps 0 a 12 são as treze remessas, na ordem recebida"
        " (jul r1 bloqueada, jul r2 reenvio).",
        color=INK_SOFT,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.965))
    path = out / "mlflow_monitoramento.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def training_figure(client: mlflow.MlflowClient, out: Path) -> Path:
    experiment = client.get_experiment_by_name(TRAINING_EXPERIMENT)
    if experiment is None:
        raise SystemExit(f"experimento {TRAINING_EXPERIMENT!r} nao encontrado; rode o pipeline com MLflow ligado")
    runs = client.search_runs([experiment.experiment_id], order_by=["attributes.start_time DESC"])
    parent = next(r for r in runs if "mlflow.parentRunId" not in r.data.tags)
    children = sorted(
        (r for r in runs if r.data.tags.get("mlflow.parentRunId") == parent.info.run_id),
        key=lambda r: r.data.metrics.get("val_roc_auc", 0),
    )
    names = {"logreg": "Regressão Logística", "xgboost": "XGBoost"}
    metrics = [
        ("val_roc_auc", "ROC AUC"),
        ("val_ks", "KS"),
        ("val_pr_auc", "PR AUC"),
        ("val_brier", "Brier (menor = melhor)"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(11.5, 3.1), dpi=130, sharey=True)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"MLflow · experimento {TRAINING_EXPERIMENT} · run {parent.info.run_name}"
        " · candidatos na validação (um run filho por candidato)",
        color=INK,
        fontsize=10.5,
        x=0.03,
        ha="left",
        y=0.99,
    )
    colors = {"xgboost": BLUE_DARK, "logreg": GRAY}
    for ax, (metric, title) in zip(axes, metrics, strict=False):
        style_axes(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        ax.set_title(title, loc="left", fontsize=10)
        values = [r.data.metrics[metric] for r in children]
        labels = [names.get(r.info.run_name, r.info.run_name) for r in children]
        bars = ax.barh(labels, values, color=[colors.get(r.info.run_name, BLUE) for r in children], height=0.55)
        for bar, value in zip(bars, values, strict=False):
            ax.text(
                value + max(values) * 0.015,
                bar.get_y() + bar.get_height() / 2,
                br(value),
                va="center",
                color=INK,
                fontsize=8.5,
            )
        ax.set_xlim(0, max(values) * 1.25)
        ax.tick_params(axis="y", labelcolor=INK, labelsize=9)
    test_auc = parent.data.metrics.get("test_roc_auc")
    note = "Selecionado: XGBoost (maior ROC AUC na validação)"
    if test_auc is not None:
        note += f"; no teste, ROC AUC {br(test_auc)}, registrado no run pai e no Model Registry"
        note += " como credit-scoring-baseline."
    fig.text(0.03, 0.02, note, color=INK_SOFT, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 0.99, 0.92))
    path = out / "mlflow_treino.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tracking-uri", default=os.environ.get("CREDITWATCH_MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    )
    parser.add_argument("--out", type=Path, default=Path("docs/img"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(args.tracking_uri)
    client = mlflow.MlflowClient()
    for path in (monitoring_figure(client, args.out), training_figure(client, args.out)):
        print(path)


if __name__ == "__main__":
    main()
