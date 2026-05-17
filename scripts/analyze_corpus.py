from __future__ import annotations

import argparse
import csv
from pathlib import Path

from nicolasm.corpus_stats import corpus_summary


FIELDNAMES = [
    "name",
    "path",
    "num_chars",
    "vocab_size",
    "H1",
    "H2",
    "H3",
    "conditional_bigram_entropy",
    "distinct_2",
    "distinct_3",
    "distinct_4",
    "gzip_compression_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute empirical diversity metrics for text corpora."
    )

    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=Path,
        help="Path to a UTF-8 text corpus. Can be passed multiple times.",
    )

    parser.add_argument(
        "--name",
        action="append",
        help=(
            "Optional corpus name. If provided, pass once per --input in the "
            "same order."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV path. If omitted, CSV is printed to stdout.",
    )

    return parser.parse_args()


def build_rows(paths: list[Path], names: list[str] | None) -> list[dict[str, str]]:
    if names is not None and len(names) != len(paths):
        raise ValueError("--name must be provided once per --input.")

    rows: list[dict[str, str]] = []

    for index, path in enumerate(paths):
        if not path.exists():
            raise FileNotFoundError(f"Could not find input file: {path}")

        text = path.read_text(encoding="utf-8")
        summary = corpus_summary(text)
        name = names[index] if names is not None else path.stem

        row: dict[str, str] = {
            "name": name,
            "path": str(path),
        }

        for key, value in summary.items():
            row[key] = str(value)

        rows.append(row)

    return rows


def write_csv(rows: list[dict[str, str]], output: Path | None) -> None:
    if output is None:
        raise ValueError("output must not be None.")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def print_csv(rows: list[dict[str, str]]) -> None:
    import sys

    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = build_rows(paths=args.input, names=args.name)

    if args.output is None:
        print_csv(rows)
    else:
        write_csv(rows, output=args.output)


if __name__ == "__main__":
    main()
