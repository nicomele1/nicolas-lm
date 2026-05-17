from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

from nicolasm.corpus_stats import corpus_summary


CORPUS_FIELDS = [
    "corpus_name",
    "corpus_path",
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

EVAL_FIELDS = [
    "model_name",
    "checkpoint_path",
    "train_tokens",
    "val_tokens",
    "test_tokens",
    "block_size",
    "embedding_dim",
    "num_heads",
    "num_layers",
    "max_steps",
    "batch_size",
    "eval_iters",
    "train_loss",
    "val_loss",
    "test_loss_mean",
    "test_loss_sd",
    "test_loss_se",
    "test_loss_ci_low",
    "test_loss_ci_high",
    "test_ppl",
    "generalization_gap",
]

OUTPUT_FIELDS = CORPUS_FIELDS + EVAL_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the effective-tokens experiment over equal-size text corpora."
        )
    )

    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=Path,
        help="Path to a UTF-8 corpus. Pass once per corpus.",
    )
    parser.add_argument(
        "--name",
        action="append",
        help="Optional corpus name. If provided, pass once per --input.",
    )
    parser.add_argument(
        "--model-name",
        action="append",
        choices=["bigram", "transformer", "llama"],
        help="Model to run. Can be passed multiple times. Default: transformer.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/effective_tokens.csv"),
        help="Final CSV path with corpus metrics and evaluation results.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("experiments/effective_tokens"),
        help="Directory for prepared equal-size corpora and intermediate CSVs.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("experiments/runs/effective_tokens"),
        help="Directory for experiment checkpoints.",
    )
    parser.add_argument(
        "--truncate-to-min-chars",
        action="store_true",
        help="Truncate all corpora to the shortest input length.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        help="Optional raw character budget. Corpora are truncated to this size.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and only run evaluation from existing checkpoints.",
    )
    parser.add_argument(
        "--skip-evaluate",
        action="store_true",
        help="Skip evaluation after training.",
    )

    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=15000)
    parser.add_argument("--eval-interval", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--eval-iters", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=32)

    return parser.parse_args()


def corpus_names(paths: list[Path], names: list[str] | None) -> list[str]:
    if names is not None and len(names) != len(paths):
        raise ValueError("--name must be provided once per --input.")

    if names is None:
        names = [path.stem for path in paths]

    if len(set(names)) != len(names):
        raise ValueError("Corpus names must be unique.")

    return names


def read_corpora(paths: list[Path]) -> list[str]:
    texts: list[str] = []

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Could not find corpus: {path}")

        texts.append(path.read_text(encoding="utf-8"))

    return texts


def target_char_count(
    texts: list[str],
    truncate_to_min_chars: bool,
    max_chars: int | None,
) -> int:
    lengths = [len(text) for text in texts]

    if not lengths:
        raise ValueError("At least one corpus is required.")

    if any(length == 0 for length in lengths):
        raise ValueError("Corpora must be nonempty.")

    target = min(lengths)

    if max_chars is not None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive.")

        if any(length < max_chars for length in lengths):
            raise ValueError("Every corpus must have at least max_chars.")

        target = max_chars
        truncate_to_min_chars = True

    if truncate_to_min_chars:
        return target

    if len(set(lengths)) != 1:
        raise ValueError(
            "Corpora must have the same raw character count. Use "
            "--truncate-to-min-chars or --max-chars to prepare equal-size "
            "copies."
        )

    return lengths[0]


def validate_split_lengths(
    num_chars: int,
    train_fraction: float,
    val_fraction: float,
    block_size: int,
) -> None:
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    train_end = int(num_chars * train_fraction)
    val_end = int(num_chars * (train_fraction + val_fraction))

    split_lengths = {
        "train": train_end,
        "validation": val_end - train_end,
        "test": num_chars - val_end,
    }

    too_short = [
        name
        for name, split_length in split_lengths.items()
        if split_length <= block_size
    ]

    if too_short:
        raise ValueError(
            "Every split must have length greater than block_size. "
            f"Too short: {', '.join(too_short)}. "
            "Use more text, a smaller block size, or larger split fractions."
        )


def prepare_corpora(
    paths: list[Path],
    names: list[str],
    texts: list[str],
    target_chars: int,
    work_dir: Path,
) -> list[Path]:
    corpus_dir = work_dir / "corpora"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    prepared_paths: list[Path] = []

    for path, name, text in zip(paths, names, texts):
        prepared_text = text[:target_chars]
        prepared_path = corpus_dir / f"{name}.txt"
        prepared_path.write_text(prepared_text, encoding="utf-8")
        prepared_paths.append(prepared_path)

        if len(prepared_text) != target_chars:
            raise RuntimeError(f"Failed to prepare equal-size corpus from {path}.")

    return prepared_paths


def command_env() -> dict[str, str]:
    env = os.environ.copy()
    current = env.get("PYTHONPATH")
    src = str(Path("src").resolve())
    env["PYTHONPATH"] = src if not current else f"{src}{os.pathsep}{current}"
    return env


def run_command(command: list[str], env: dict[str, str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def train_model(
    corpus_path: Path,
    corpus_name: str,
    model_name: str,
    args: argparse.Namespace,
    env: dict[str, str],
) -> None:
    run_command(
        [
            sys.executable,
            "scripts/train.py",
            "--model-name",
            model_name,
            "--data-path",
            str(corpus_path),
            "--runs-dir",
            str(args.runs_dir / corpus_name),
            "--block-size",
            str(args.block_size),
            "--batch-size",
            str(args.batch_size),
            "--max-steps",
            str(args.max_steps),
            "--eval-interval",
            str(args.eval_interval),
            "--learning-rate",
            str(args.learning_rate),
            "--embedding-dim",
            str(args.embedding_dim),
            "--num-heads",
            str(args.num_heads),
            "--num-layers",
            str(args.num_layers),
            "--dropout",
            str(args.dropout),
            "--train-fraction",
            str(args.train_fraction),
            "--val-fraction",
            str(args.val_fraction),
        ],
        env=env,
    )


def evaluate_model(
    corpus_path: Path,
    corpus_name: str,
    model_name: str,
    args: argparse.Namespace,
    eval_csv: Path,
    env: dict[str, str],
) -> None:
    run_command(
        [
            sys.executable,
            "scripts/evaluate.py",
            "--model-name",
            model_name,
            "--data-path",
            str(corpus_path),
            "--runs-dir",
            str(args.runs_dir / corpus_name),
            "--eval-iters",
            str(args.eval_iters),
            "--batch-size",
            str(args.eval_batch_size),
            "--output",
            str(eval_csv),
        ],
        env=env,
    )


def read_single_eval_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one evaluation row in {path}.")

    return rows[0]


def write_output(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    names = corpus_names(args.input, args.name)
    model_names = args.model_name or ["transformer"]
    texts = read_corpora(args.input)
    target_chars = target_char_count(
        texts=texts,
        truncate_to_min_chars=args.truncate_to_min_chars,
        max_chars=args.max_chars,
    )
    validate_split_lengths(
        num_chars=target_chars,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        block_size=args.block_size,
    )
    prepared_paths = prepare_corpora(
        paths=args.input,
        names=names,
        texts=texts,
        target_chars=target_chars,
        work_dir=args.work_dir,
    )
    env = command_env()
    rows: list[dict[str, str]] = []

    for corpus_name, corpus_path in zip(names, prepared_paths):
        text = corpus_path.read_text(encoding="utf-8")
        summary = corpus_summary(text)
        corpus_row = {
            "corpus_name": corpus_name,
            "corpus_path": str(corpus_path),
            **{key: str(value) for key, value in summary.items()},
        }

        for model_name in model_names:
            if not args.skip_train:
                train_model(
                    corpus_path=corpus_path,
                    corpus_name=corpus_name,
                    model_name=model_name,
                    args=args,
                    env=env,
                )

            if args.skip_evaluate:
                continue

            eval_csv = args.work_dir / "eval" / f"{corpus_name}_{model_name}.csv"
            if eval_csv.exists():
                eval_csv.unlink()

            evaluate_model(
                corpus_path=corpus_path,
                corpus_name=corpus_name,
                model_name=model_name,
                args=args,
                eval_csv=eval_csv,
                env=env,
            )
            eval_row = read_single_eval_row(eval_csv)

            row = {
                **corpus_row,
                **{field: eval_row.get(field, "") for field in EVAL_FIELDS},
            }
            rows.append(row)

    if rows:
        write_output(rows, args.output)
        print(f"Saved effective-tokens results to {args.output}")


if __name__ == "__main__":
    main()
