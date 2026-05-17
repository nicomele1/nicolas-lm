from __future__ import annotations

import argparse
import csv
from pathlib import Path


CORPUS_COLUMNS = [
    "corpus_name",
    "num_chars",
    "vocab_size",
    "H1",
    "H2",
    "H3",
    "conditional_bigram_entropy",
    "distinct_4",
    "gzip_compression_ratio",
]

RESULT_COLUMNS = [
    "model_name",
    "corpus_name",
    "test_loss_mean",
    "test_loss_se",
    "test_ppl",
    "generalization_gap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create compact CSV and Markdown summaries of experiment results."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Effective-tokens CSV produced by scripts/run_effective_tokens.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/reports"),
        help="Directory for summary CSV and Markdown files.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def format_number(value: str, digits: int = 4) -> str:
    if value == "":
        return ""

    number = float(value)

    if abs(number) >= 100:
        return f"{number:.2f}"

    return f"{number:.{digits}f}"


def corpus_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    output: list[dict[str, str]] = []

    for row in rows:
        corpus_name = row["corpus_name"]
        if corpus_name in seen:
            continue

        seen.add(corpus_name)
        output.append(
            {
                "corpus_name": corpus_name,
                "num_chars": row["num_chars"],
                "vocab_size": row["vocab_size"],
                "H1": format_number(row["H1"]),
                "H2": format_number(row["H2"]),
                "H3": format_number(row["H3"]),
                "conditional_bigram_entropy": format_number(
                    row["conditional_bigram_entropy"]
                ),
                "distinct_4": format_number(row["distinct_4"]),
                "gzip_compression_ratio": format_number(
                    row["gzip_compression_ratio"]
                ),
            }
        )

    return output


def result_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []

    for row in rows:
        output.append(
            {
                "model_name": row["model_name"],
                "corpus_name": row["corpus_name"],
                "test_loss_mean": format_number(row["test_loss_mean"]),
                "test_loss_se": format_number(row["test_loss_se"]),
                "test_ppl": format_number(row["test_ppl"]),
                "generalization_gap": format_number(row["generalization_gap"]),
            }
        )

    return output


def effect_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_model: dict[str, dict[str, dict[str, str]]] = {}

    for row in rows:
        by_model.setdefault(row["model_name"], {})[row["corpus_name"]] = row

    output: list[dict[str, str]] = []

    for model_name, model_rows in sorted(by_model.items()):
        if "medium" not in model_rows or "high" not in model_rows:
            continue

        medium = model_rows["medium"]
        high = model_rows["high"]
        delta_test = float(high["test_loss_mean"]) - float(medium["test_loss_mean"])
        relative_ppl_change = (
            float(high["test_ppl"]) - float(medium["test_ppl"])
        ) / float(medium["test_ppl"])
        delta_gap = float(high["generalization_gap"]) - float(
            medium["generalization_gap"]
        )

        output.append(
            {
                "model_name": model_name,
                "delta_test_loss_high_minus_medium": format_number(str(delta_test)),
                "relative_ppl_change_high_vs_medium": format_number(
                    str(relative_ppl_change)
                ),
                "delta_gap_high_minus_medium": format_number(str(delta_gap)),
            }
        )

    return output


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]], fieldnames: list[str]) -> str:
    header = "| " + " | ".join(fieldnames) + " |"
    separator = "| " + " | ".join("---" for _ in fieldnames) + " |"
    body = [
        "| " + " | ".join(row.get(field, "") for field in fieldnames) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.input)
    corpus = corpus_rows(rows)
    results = result_rows(rows)
    effects = effect_rows(rows)

    write_csv(args.output_dir / "corpus_summary.csv", corpus, CORPUS_COLUMNS)
    write_csv(args.output_dir / "model_results_summary.csv", results, RESULT_COLUMNS)
    write_csv(
        args.output_dir / "effect_sizes_summary.csv",
        effects,
        [
            "model_name",
            "delta_test_loss_high_minus_medium",
            "relative_ppl_change_high_vs_medium",
            "delta_gap_high_minus_medium",
        ],
    )

    report = "\n\n".join(
        [
            "# Experiment Summary",
            "## Corpus Metrics",
            markdown_table(corpus, CORPUS_COLUMNS),
            "## Model Results",
            markdown_table(results, RESULT_COLUMNS),
            "## Medium vs High Effect Sizes",
            markdown_table(
                effects,
                [
                    "model_name",
                    "delta_test_loss_high_minus_medium",
                    "relative_ppl_change_high_vs_medium",
                    "delta_gap_high_minus_medium",
                ],
            ),
        ]
    )

    (args.output_dir / "experiment_summary.md").write_text(
        report + "\n",
        encoding="utf-8",
    )

    print(f"Wrote summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
