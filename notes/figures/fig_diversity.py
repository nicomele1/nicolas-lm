"""Generate the corpus-comparison figure from the versioned result CSV."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/results/effective_tokens_1M.csv"
OUTPUT = Path(__file__).with_suffix(".pdf")


def corpus_rows() -> dict[str, dict[str, str]]:
    with RESULTS.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["corpus_name"]: row for row in rows}


def main() -> None:
    rows = corpus_rows()
    medium, high = rows["medium"], rows["high"]

    entropy_labels = [r"$H_1$", r"$H_2$", r"$H_3$"]
    entropy_delta = [float(high[key]) - float(medium[key]) for key in ("H1", "H2", "H3")]
    ratio_labels = [r"$D_4$", r"$\rho_{\mathrm{gzip}}$"]
    ratio_delta_pp = [
        100 * (float(high[key]) - float(medium[key]))
        for key in ("distinct_4", "gzip_compression_ratio")
    ]

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), constrained_layout=True)
    color = "#315b7d"

    for ax, labels, values, xlabel in (
        (axes[0], entropy_labels, entropy_delta, "High − Medium (nats)"),
        (axes[1], ratio_labels, ratio_delta_pp, "High − Medium (percentage points)"),
    ):
        positions = range(len(labels))
        ax.axvline(0, color="black", linewidth=0.7)
        ax.scatter(values, positions, color=color, s=28, zorder=3)
        for value, position in zip(values, positions):
            ax.hlines(position, 0, value, color=color, linewidth=1.2)
            ax.annotate(f"{value:.3f}", (value, position), xytext=(5, 0),
                        textcoords="offset points", va="center", fontsize=8)
        ax.set_yticks(list(positions), labels)
        ax.set_xlabel(xlabel)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="0.9", linewidth=0.6)

    fig.savefig(OUTPUT, bbox_inches="tight")


if __name__ == "__main__":
    main()
