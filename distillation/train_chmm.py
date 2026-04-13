"""Train a CHMM distilled from sampled LLM outputs.

This script mirrors the existing `train_hmm.py` workflow but uses a
large-vocabulary CHMM with a configurable clone schedule.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from tqdm import tqdm

from chmm import CHMM, build_clone_schedule, rank_tokens_from_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--checkpoint", default=0, type=int)
    parser.add_argument("--save_per_step", default=1, type=int)
    parser.add_argument("--init_only", action="store_true")

    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--dev_file", default="", type=str)
    parser.add_argument("--total_chunks", required=True, type=int)
    parser.add_argument("--sample_length", default=None, type=int)

    parser.add_argument("--tokenizer_name_or_path", default="", type=str)
    parser.add_argument("--vocab_size", default=None, type=int)
    parser.add_argument("--eos_token_id", default=None, type=int)

    parser.add_argument("--batch_size", default=2048, type=int)
    parser.add_argument("--train_eval_size", default=8192, type=int)
    parser.add_argument("--max_train_batches", default=None, type=int)
    parser.add_argument("--max_dev_batches", default=None, type=int)

    parser.add_argument("--em_schedule", default="", type=str)
    parser.add_argument("--init_chunk_count", default=4, type=int)
    parser.add_argument("--init_context", default="both", choices=["prev", "next", "both"])

    parser.add_argument("--clone_top4", default=64, type=int)
    parser.add_argument("--clone_top2", default=256, type=int)

    parser.add_argument("--storage_dtype", default="bfloat16", type=str)
    parser.add_argument("--pseudocount", default=1e-3, type=float)
    parser.add_argument("--clone_pseudocount", default=None, type=float)
    parser.add_argument("--initial_pseudocount", default=None, type=float)
    parser.add_argument("--online_count_decay", default=0.0, type=float)
    parser.add_argument("--row_chunk_size", default=512, type=int)
    parser.add_argument("--device", default="cuda", type=str)

    parser.add_argument("--log_file", default="", type=str)

    return parser.parse_args()


def resolve_vocab_and_eos(args: argparse.Namespace) -> tuple[int, int]:
    if args.vocab_size is not None and args.eos_token_id is not None:
        return int(args.vocab_size), int(args.eos_token_id)

    if not args.tokenizer_name_or_path:
        raise ValueError(
            "Provide either (--vocab_size and --eos_token_id) or "
            "--tokenizer_name_or_path."
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    return int(tokenizer.vocab_size), int(tokenizer.eos_token_id)


def parse_em_schedule(schedule: str, total_chunks: int) -> list[tuple[int, int]]:
    if not schedule:
        return [(1, total_chunks)]
    return [
        tuple(int(y) for y in item.split(","))
        for item in schedule.split(";")
        if item.strip()
    ]


def chunk_file(data_path: str, dataset: str, chunk_id: int, total_chunks: int) -> str:
    if total_chunks == 1:
        return f"{data_path}/{dataset}.train"
    return f"{data_path}/{dataset}.train.{chunk_id}"


def load_sequences(path: str, sample_length: int | None) -> torch.Tensor:
    seqs = torch.load(path, map_location="cpu", weights_only=True).long()
    if sample_length is not None:
        seqs = seqs[:, :sample_length]
    return seqs.contiguous()


def trim_lengths(input_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    eos_mask = input_ids.eq(eos_token_id)
    has_eos = eos_mask.any(dim=1)
    first_eos = torch.argmax(eos_mask.int(), dim=1)
    full_length = torch.full_like(first_eos, input_ids.shape[1] - 1)
    first_eos = torch.where(has_eos, first_eos, full_length)
    return first_eos + 1


def iter_length_buckets(
    input_ids: torch.Tensor,
    eos_token_id: int,
) -> Iterable[torch.Tensor]:
    lengths = trim_lengths(input_ids, eos_token_id)
    for length in torch.unique(lengths, sorted=True).tolist():
        bucket = input_ids[lengths == length, :length]
        if bucket.numel() > 0:
            yield bucket.contiguous()


def count_tokens_for_schedule(
    input_ids: torch.Tensor,
    vocab_size: int,
    eos_token_id: int,
) -> torch.Tensor:
    lengths = trim_lengths(input_ids, eos_token_id)
    positions = torch.arange(input_ids.shape[1])[None, :]
    valid = positions < lengths[:, None]
    return torch.bincount(input_ids[valid], minlength=vocab_size)


def take_train_eval_subset(
    data_path: str,
    dataset: str,
    sample_length: int | None,
    size: int,
    total_chunks: int,
) -> torch.Tensor:
    chunk0 = load_sequences(chunk_file(data_path, dataset, 0, total_chunks), sample_length)
    return chunk0[:size].clone()


def evaluate_model(
    model: CHMM,
    input_ids: torch.Tensor,
    batch_size: int,
    eos_token_id: int,
    max_batches: int | None = None,
) -> dict[str, float]:
    total_ll = 0.0
    total_sequences = 0
    total_tokens = 0
    num_batches = 0

    with torch.no_grad():
        for start in range(0, input_ids.shape[0], batch_size):
            batch = input_ids[start : start + batch_size]
            for bucket in iter_length_buckets(batch, eos_token_id):
                ll, _ = model.forward_backward(bucket, return_messages=False)
                total_ll += float(ll.sum().item())
                total_sequences += int(bucket.shape[0])
                total_tokens += int(bucket.numel())
            num_batches += 1
            if max_batches is not None and num_batches >= max_batches:
                break

    return {
        "nll_per_sequence": -total_ll / max(total_sequences, 1),
        "nll_per_token": -total_ll / max(total_tokens, 1),
        "num_sequences": total_sequences,
        "num_tokens": total_tokens,
    }


def estimate_memory_gib(model: CHMM) -> dict[str, float]:
    dtype_size = torch.tensor([], dtype=model.token_log_probs.dtype).element_size()
    token_table = model.hidden_states * model.vocab_size * dtype_size / (1024 ** 3)
    token_counts = model.hidden_states * model.vocab_size * 4 / (1024 ** 3)
    clone_table = model.clone_log_probs.numel() * dtype_size / (1024 ** 3)
    clone_counts = model.clone_log_probs.numel() * 4 / (1024 ** 3)
    return {
        "token_table_gib": token_table,
        "token_counts_gib": token_counts,
        "clone_table_gib": clone_table,
        "clone_counts_gib": clone_counts,
    }


def initialize_checkpoint_zero(
    args: argparse.Namespace,
    vocab_size: int,
    eos_token_id: int,
    device: torch.device,
) -> CHMM:
    print("Initializing checkpoint-0...")

    init_chunk_count = min(args.init_chunk_count, args.total_chunks)
    token_frequency = torch.zeros(vocab_size, dtype=torch.long)

    for chunk_id in range(init_chunk_count):
        path = chunk_file(args.data_path, args.dataset, chunk_id, args.total_chunks)
        seqs = load_sequences(path, args.sample_length)
        token_frequency += count_tokens_for_schedule(seqs, vocab_size, eos_token_id)

    ranked_tokens = rank_tokens_from_counts(
        token_frequency,
        protected_token_ids=[eos_token_id],
    )
    clones_per_token = build_clone_schedule(
        vocab_size=vocab_size,
        ranked_token_ids=ranked_tokens,
        four_clone_tokens=args.clone_top4,
        two_clone_tokens=args.clone_top2,
        protected_token_ids=[eos_token_id],
    )

    model = CHMM(
        vocab_size=vocab_size,
        eos_token_id=eos_token_id,
        clones_per_token=clones_per_token,
        storage_dtype=args.storage_dtype,
        device=device,
    )

    memory = estimate_memory_gib(model)
    print(
        f"CHMM states={model.hidden_states}, max_clones={model.max_clones}, "
        f"token_table={memory['token_table_gib']:.2f} GiB, "
        f"token_counts={memory['token_counts_gib']:.2f} GiB"
    )

    token_counts, clone_counts, initial_counts = model.empty_count_buffers(device=device)

    for chunk_id in range(init_chunk_count):
        path = chunk_file(args.data_path, args.dataset, chunk_id, args.total_chunks)
        seqs = load_sequences(path, args.sample_length)

        for start in tqdm(
            range(0, seqs.shape[0], args.batch_size),
            desc=f"init chunk {chunk_id}",
        ):
            batch = seqs[start : start + args.batch_size]
            for bucket in iter_length_buckets(batch, eos_token_id):
                model.accumulate_hard_counts(
                    bucket,
                    token_counts,
                    clone_counts,
                    initial_counts,
                    context_mode=args.init_context,
                )

    model.update_from_counts(
        token_counts=token_counts,
        clone_counts=clone_counts,
        initial_counts=initial_counts,
        pseudocount=args.pseudocount,
        clone_pseudocount=args.clone_pseudocount,
        initial_pseudocount=args.initial_pseudocount,
        row_chunk_size=args.row_chunk_size,
    )

    ckpt_dir = Path(args.model_path) / "checkpoint-0"
    model.save_pretrained(ckpt_dir)
    print(f"Saved checkpoint-0 to {ckpt_dir}")
    return model


def expectation_step(
    model: CHMM,
    chunk_paths: list[str],
    args: argparse.Namespace,
    eos_token_id: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], dict[str, float]]:
    token_counts, clone_counts, initial_counts = model.empty_count_buffers(device=model.device)

    total_ll = 0.0
    total_sequences = 0
    total_tokens = 0
    seen_batches = 0

    for path in chunk_paths:
        seqs = load_sequences(path, args.sample_length)
        for start in tqdm(range(0, seqs.shape[0], args.batch_size), desc=f"em {Path(path).name}"):
            batch = seqs[start : start + args.batch_size]
            for bucket in iter_length_buckets(batch, eos_token_id):
                ll = model.accumulate_expected_counts(
                    bucket,
                    token_counts,
                    clone_counts,
                    initial_counts,
                )
                total_ll += float(ll.sum().item())
                total_sequences += int(bucket.shape[0])
                total_tokens += int(bucket.numel())
            seen_batches += 1
            if args.max_train_batches is not None and seen_batches >= args.max_train_batches:
                break
        if args.max_train_batches is not None and seen_batches >= args.max_train_batches:
            break

    return (token_counts, clone_counts, initial_counts), {
        "nll_per_sequence": -total_ll / max(total_sequences, 1),
        "nll_per_token": -total_ll / max(total_tokens, 1),
        "num_sequences": total_sequences,
        "num_tokens": total_tokens,
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.model_path, exist_ok=True)

    if args.log_file:
        os.makedirs(str(Path(args.log_file).parent), exist_ok=True)
        with open(args.log_file, "a+", encoding="utf-8") as fout:
            fout.write(json.dumps(vars(args), sort_keys=True) + "\n")

    vocab_size, eos_token_id = resolve_vocab_and_eos(args)
    schedule = parse_em_schedule(args.em_schedule, args.total_chunks)
    if not 0.0 <= args.online_count_decay < 1.0:
        raise ValueError("--online_count_decay must be in [0, 1).")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(args.device)

    checkpoint_dir = Path(args.model_path) / f"checkpoint-{args.checkpoint}"
    if checkpoint_dir.exists():
        model = CHMM.from_pretrained(checkpoint_dir, map_location=device)
    else:
        if args.checkpoint != 0:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")
        model = initialize_checkpoint_zero(args, vocab_size, eos_token_id, device)

    model = model.to(device)
    if args.init_only:
        return

    dev_path = args.dev_file or f"{args.data_path}/{args.dataset}.dev"
    dev_data = load_sequences(dev_path, args.sample_length)
    train_eval = take_train_eval_subset(
        args.data_path,
        args.dataset,
        args.sample_length,
        args.train_eval_size,
        args.total_chunks,
    )

    if args.online_count_decay > 0.0 and args.checkpoint != 0:
        print(
            "Warning: online_count_decay resumes EMA counts from the current checkpoint "
            "parameters only; historical count buffers are not restored."
        )

    step_offset = args.checkpoint
    ema_token_counts: torch.Tensor | None = None
    ema_clone_counts: torch.Tensor | None = None
    ema_initial_counts: torch.Tensor | None = None

    for step_count, step_size in schedule:
        for _ in range(step_count):
            started = time.time()
            chunk_paths = [
                chunk_file(args.data_path, args.dataset, idx % args.total_chunks, args.total_chunks)
                for idx in range(step_offset, step_offset + step_size)
            ]

            (step_token_counts, step_clone_counts, step_initial_counts), train_metrics = expectation_step(
                model,
                chunk_paths,
                args,
                eos_token_id,
            )

            if args.online_count_decay > 0.0:
                if ema_token_counts is None:
                    ema_token_counts = step_token_counts
                    ema_clone_counts = step_clone_counts
                    ema_initial_counts = step_initial_counts
                else:
                    decay = args.online_count_decay
                    keep = 1.0 - decay
                    ema_token_counts.mul_(decay).add_(step_token_counts, alpha=keep)
                    ema_clone_counts.mul_(decay).add_(step_clone_counts, alpha=keep)
                    ema_initial_counts.mul_(decay).add_(step_initial_counts, alpha=keep)

                token_counts = ema_token_counts
                clone_counts = ema_clone_counts
                initial_counts = ema_initial_counts
            else:
                token_counts = step_token_counts
                clone_counts = step_clone_counts
                initial_counts = step_initial_counts

            model.update_from_counts(
                token_counts=token_counts,
                clone_counts=clone_counts,
                initial_counts=initial_counts,
                pseudocount=args.pseudocount,
                clone_pseudocount=args.clone_pseudocount,
                initial_pseudocount=args.initial_pseudocount,
                row_chunk_size=args.row_chunk_size,
            )

            dev_metrics = evaluate_model(
                model,
                dev_data,
                args.batch_size,
                eos_token_id,
                max_batches=args.max_dev_batches,
            )
            train_eval_metrics = evaluate_model(
                model,
                train_eval,
                args.batch_size,
                eos_token_id,
                max_batches=args.max_dev_batches,
            )

            ckpt = step_offset + step_size
            elapsed = time.time() - started
            msg = (
                f"ckpt={ckpt}\t"
                f"train_step_nll_tok={train_metrics['nll_per_token']:.6f}\t"
                f"train_eval_nll_tok={train_eval_metrics['nll_per_token']:.6f}\t"
                f"dev_nll_tok={dev_metrics['nll_per_token']:.6f}\t"
                f"elapsed_s={elapsed:.1f}"
            )
            print(msg)
            if args.log_file:
                with open(args.log_file, "a+", encoding="utf-8") as fout:
                    fout.write(msg + "\n")

            if ckpt % args.save_per_step == 0:
                out_dir = Path(args.model_path) / f"checkpoint-{ckpt}"
                model.save_pretrained(out_dir)

            step_offset += step_size


if __name__ == "__main__":
    main()
