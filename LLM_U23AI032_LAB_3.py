import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers


TRAIN_SIZES = [100_000, 300_000, 500_000, 1_000_000]
VOCAB_SIZES = [20_000, 30_000, 50_000]
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def read_sentences(csv_path: Path) -> list[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")

    sentences = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            text = (row.get("text") or "").strip()
            if text:
                sentences.append(text)
    return sentences


def build_tokenizer(algorithm: str, vocab_size: int, min_frequency: int) -> tuple[Tokenizer, object]:
    if algorithm == "bpe":
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
        )
    elif algorithm == "wordpiece":
        tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return tokenizer, trainer


def summarize_tokenization(tokenizer: Tokenizer, sentences: list[str]) -> dict:
    sentence_token_counts = []
    token_lengths = []
    total_tokens = 0
    total_chars = 0
    unk_count = 0
    unique_tokens = set()

    for sentence in sentences:
        encoded = tokenizer.encode(sentence)
        tokens = encoded.tokens

        token_count = len(tokens)
        sentence_token_counts.append(token_count)
        total_tokens += token_count
        total_chars += len(sentence)

        for tok in tokens:
            unique_tokens.add(tok)
            token_lengths.append(len(tok))
            if tok == "[UNK]":
                unk_count += 1

    avg_tokens_per_sentence = total_tokens / max(1, len(sentences))
    avg_chars_per_token = total_chars / max(1, total_tokens)
    unk_rate = unk_count / max(1, total_tokens)

    return {
        "num_sentences": len(sentences),
        "total_tokens": total_tokens,
        "avg_tokens_per_sentence": avg_tokens_per_sentence,
        "avg_chars_per_token": avg_chars_per_token,
        "unk_rate": unk_rate,
        "unique_tokens_in_split": len(unique_tokens),
        "mean_token_surface_length": mean(token_lengths) if token_lengths else 0.0,
        "min_tokens_per_sentence": min(sentence_token_counts) if sentence_token_counts else 0,
        "max_tokens_per_sentence": max(sentence_token_counts) if sentence_token_counts else 0,
    }


def save_samples(
    tokenizer: Tokenizer,
    sentences: list[str],
    out_path: Path,
    sample_count: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # A simple spread: first N and last N/2 if available to include shorter and longer cases.
    picks = []
    head_n = min(sample_count, len(sentences))
    picks.extend(list(range(head_n)))

    tail_n = min(sample_count // 2, len(sentences))
    if tail_n > 0:
        picks.extend(list(range(max(0, len(sentences) - tail_n), len(sentences))))

    # Deduplicate while preserving order.
    seen = set()
    unique_picks = []
    for idx in picks:
        if idx not in seen:
            unique_picks.append(idx)
            seen.add(idx)

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_index", "text", "tokens", "token_count"],
        )
        writer.writeheader()

        for idx in unique_picks:
            text = sentences[idx]
            tokens = tokenizer.encode(text).tokens
            writer.writerow(
                {
                    "sample_index": idx,
                    "text": text,
                    "tokens": " | ".join(tokens),
                    "token_count": len(tokens),
                }
            )


def run_experiments(
    data_dir: Path,
    output_dir: Path,
    train_sizes: list[int],
    vocab_sizes: list[int],
    min_frequency: int,
    sample_count: int,
) -> None:
    dev_sentences = read_sentences(data_dir / "dev.csv")
    test_sentences = read_sentences(data_dir / "test.csv")

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizers_dir = output_dir / "tokenizers"
    samples_dir = output_dir / "samples"

    all_rows = []
    full_report = []

    for algorithm in ("bpe", "wordpiece"):
        for train_size in train_sizes:
            train_path = data_dir / f"train_{train_size}.csv"
            train_sentences = read_sentences(train_path)

            for vocab_size in vocab_sizes:
                tokenizer, trainer = build_tokenizer(algorithm, vocab_size, min_frequency)
                tokenizer.train_from_iterator(train_sentences, trainer=trainer)

                model_id = f"{algorithm}_train{train_size}_vocab{vocab_size}"
                save_path = tokenizers_dir / algorithm / f"train_{train_size}_vocab_{vocab_size}"
                save_path.mkdir(parents=True, exist_ok=True)
                tokenizer.save(str(save_path / "tokenizer.json"))

                dev_metrics = summarize_tokenization(tokenizer, dev_sentences)
                test_metrics = summarize_tokenization(tokenizer, test_sentences)

                all_rows.append(
                    {
                        "algorithm": algorithm,
                        "train_size": train_size,
                        "vocab_size": vocab_size,
                        "dev_avg_tokens_per_sentence": dev_metrics["avg_tokens_per_sentence"],
                        "test_avg_tokens_per_sentence": test_metrics["avg_tokens_per_sentence"],
                        "dev_avg_chars_per_token": dev_metrics["avg_chars_per_token"],
                        "test_avg_chars_per_token": test_metrics["avg_chars_per_token"],
                        "dev_unk_rate": dev_metrics["unk_rate"],
                        "test_unk_rate": test_metrics["unk_rate"],
                        "dev_unique_tokens_in_split": dev_metrics["unique_tokens_in_split"],
                        "test_unique_tokens_in_split": test_metrics["unique_tokens_in_split"],
                    }
                )

                full_report.append(
                    {
                        "algorithm": algorithm,
                        "train_size": train_size,
                        "vocab_size": vocab_size,
                        "tokenizer_path": str((save_path / "tokenizer.json").resolve()),
                        "dev": dev_metrics,
                        "test": test_metrics,
                    }
                )

                save_samples(
                    tokenizer,
                    dev_sentences,
                    samples_dir / f"{model_id}_dev_samples.csv",
                    sample_count,
                )
                save_samples(
                    tokenizer,
                    test_sentences,
                    samples_dir / f"{model_id}_test_samples.csv",
                    sample_count,
                )

    with (output_dir / "lab3_tokenization_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "algorithm",
                "train_size",
                "vocab_size",
                "dev_avg_tokens_per_sentence",
                "test_avg_tokens_per_sentence",
                "dev_avg_chars_per_token",
                "test_avg_chars_per_token",
                "dev_unk_rate",
                "test_unk_rate",
                "dev_unique_tokens_in_split",
                "test_unique_tokens_in_split",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    with (output_dir / "lab3_tokenization_report.json").open("w", encoding="utf-8") as handle:
        json.dump(full_report, handle, indent=2, ensure_ascii=False)

    print("LAB 3 experiments complete.")
    print(f"Results CSV: {(output_dir / 'lab3_tokenization_comparison.csv').resolve()}")
    print(f"Results JSON: {(output_dir / 'lab3_tokenization_report.json').resolve()}")


def parse_int_list(values: list[str]) -> list[int]:
    return [int(v) for v in values]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LAB 3 BPE and WordPiece experiments on prepared datasets."
    )
    parser.add_argument(
        "--data-dir",
        default="outputs/hin_Deva",
        help="Prepared LAB 3 dataset directory containing train/dev/test CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/hin_Deva/experiments",
        help="Directory to store experiment outputs.",
    )
    parser.add_argument(
        "--train-sizes",
        nargs="+",
        default=[str(x) for x in TRAIN_SIZES],
        help="Training subset sizes to evaluate.",
    )
    parser.add_argument(
        "--vocab-sizes",
        nargs="+",
        default=[str(x) for x in VOCAB_SIZES],
        help="Vocabulary sizes to evaluate.",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum pair/subword frequency while training tokenizers.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=20,
        help="How many sample sentences per split to export tokenization examples for.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    run_experiments(
        data_dir=data_dir,
        output_dir=output_dir,
        train_sizes=parse_int_list(args.train_sizes),
        vocab_sizes=parse_int_list(args.vocab_sizes),
        min_frequency=args.min_frequency,
        sample_count=args.sample_count,
    )


if __name__ == "__main__":
    main()
