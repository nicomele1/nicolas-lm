"""Generate non-redundant model-result panels from the versioned CSV."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/results/effective_tokens_1M.csv"
OUTPUT = Path(__file__).with_suffix(".pdf")
MODELS = ("transformer", "llama")
LABELS = {"transformer": "Transformer", "llama": "LLaMA-style"}
COLORS = {"medium": "#315b7d", "high": "#a23b4a"}


def indexed_rows() -> dict[tuple[str, str], dict[str, str]]:
    with RESULTS.open(encoding="utf-8", newline="") as handle:
        return {(row["model_name"], row["corpus_name"]): row for row in csv.DictReader(handle)}


def main() -> None:
    rows = indexed_rows()
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75), constrained_layout=True)

    for y, model in enumerate(MODELS):
        med = rows[(model, "medium")]
        high = rows[(model, "high")]
        for ax, field in zip(axes, ("test_loss_mean", "generalization_gap")):
            values = [float(med[field]), float(high[field])]
            ax.plot(values, [y, y], color="0.65", linewidth=1.1, zorder=1)
            for corpus, value in zip(("medium", "high"), values):
                ax.scatter(value, y, s=34, color=COLORS[corpus], zorder=2,
                           label=corpus.capitalize() if y == 0 else None)

    for ax, title, xlabel in (
        (axes[0], "Empirical test risk", "Cross-entropy (nats/character)"),
        (axes[1], "Test − train difference", "Nats/character"),
    ):
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_yticks(range(len(MODELS)), [LABELS[m] for m in MODELS])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="0.9", linewidth=0.6)

    axes[0].legend(frameon=False, loc="best")
    fig.savefig(OUTPUT, bbox_inches="tight")


if __name__ == "__main__":
    main()
