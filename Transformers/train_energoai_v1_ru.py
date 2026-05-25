"""
Sequential Russian pretraining/fine-tuning script for energoai_transformer_v1.

Default profile is conservative for one 32GB RTX 5090:
  - bf16 model parameters
  - gradient checkpointing
  - SDPA/flash attention path
  - micro batch size 1
  - 8-bit AdamW via bitsandbytes
  - streaming datasets, one dataset after another

Install on the training server:
  pip install torch datasets transformers tokenizers accelerate bitsandbytes tqdm

Example:
  python Transformers/train_energoai_v1_ru.py ^
    --tokenizer ai-forever/rugpt3small_based_on_gpt2 ^
    --output_dir runs/energoai_v1_ru ^
    --max_steps 100000 ^
    --seq_len 512
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm

try:
    from datasets import load_dataset
except ImportError as exc:
    raise SystemExit("Install datasets: pip install datasets") from exc

try:
    from transformers import AutoTokenizer
except ImportError as exc:
    raise SystemExit("Install transformers: pip install transformers") from exc

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from energoai_transformer_v1 import EnergoAI, EnergoAIConfig


DEFAULT_RU_DATASETS: List[Dict[str, Any]] = [
    {
        "path": "wikimedia/wikipedia",
        "name": "20231101.ru",
        "split": "train",
        "text_fields": ["title", "text"],
    },
    {
        "path": "IlyaGusev/gazeta",
        "split": "train",
        "text_fields": ["title", "text", "summary"],
    },
    {
        "path": "IlyaGusev/ru_turbo_alpaca",
        "split": "train",
        "text_fields": ["instruction", "input", "output"],
    },
    {
        "path": "IlyaGusev/ru_turbo_saiga",
        "split": "train",
        "text_fields": ["instruction", "input", "output"],
    },
    {
        "path": "ai-forever/ru_turbo_alpaca",
        "split": "train",
        "text_fields": ["instruction", "input", "output"],
    },
    {
        "path": "HuggingFaceFW/fineweb-2",
        "name": "rus_Cyrl",
        "split": "train",
        "text_fields": ["text"],
    },
    {
        "path": "uonlp/CulturaX",
        "name": "ru",
        "split": "train",
        "text_fields": ["text"],
    },
    {
        "path": "oscar-corpus/OSCAR-2301",
        "name": "ru",
        "split": "train",
        "text_fields": ["text"],
    },
    {
        "path": "allenai/c4",
        "name": "ru",
        "split": "train",
        "text_fields": ["text"],
    },
    {
        "path": "mc4",
        "name": "ru",
        "split": "train",
        "text_fields": ["text"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="runs/energoai_v1_ru")
    parser.add_argument("--tokenizer", type=str, default="ai-forever/rugpt3small_based_on_gpt2")
    parser.add_argument("--dataset_manifest", type=str, default="")
    parser.add_argument("--print_default_manifest", action="store_true")
    parser.add_argument("--fail_on_dataset_error", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")

    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--max_docs_per_dataset", type=int, default=100000)
    parser.add_argument("--shuffle_buffer", type=int, default=10000)
    parser.add_argument("--min_chars", type=int, default=64)
    parser.add_argument("--min_cyrillic_ratio", type=float, default=0.15)
    parser.add_argument("--append_eos", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--learning_rate", type=float, default=2.0e-4)
    parser.add_argument("--min_learning_rate", type=float, default=2.0e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--optimizer", choices=["adamw8bit", "adamw_torch"], default="adamw8bit")

    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true")

    parser.add_argument("--hidden_size", type=int, default=2048)
    parser.add_argument("--num_layers", type=int, default=24)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--num_kv_heads", type=int, default=4)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--state_dim", type=int, default=256)
    parser.add_argument("--coda_interval", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=8192)
    parser.add_argument("--mlp_intermediate_factor", type=int, default=6)
    parser.add_argument("--efficient_intermediate_factor", type=int, default=4)
    parser.add_argument("--no_flash_attention", action="store_true")

    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--keep_last_checkpoints", type=int, default=3)
    parser.add_argument("--save_optimizer", action="store_true")
    parser.add_argument("--resume", type=str, default="")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset_manifest(path: str) -> List[Dict[str, Any]]:
    if not path:
        return DEFAULT_RU_DATASETS
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset manifest must be a JSON list")
    return data


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(stringify(x) for x in value if x is not None)
    if isinstance(value, dict):
        return "\n".join(stringify(v) for v in value.values() if v is not None)
    return str(value)


def extract_text(row: Dict[str, Any], spec: Dict[str, Any]) -> str:
    fields = spec.get("text_fields") or []
    parts = [stringify(row.get(name)).strip() for name in fields if row.get(name) is not None]
    parts = [p for p in parts if p]
    if parts:
        return "\n\n".join(parts)

    fallback_groups = [
        ["text"],
        ["content"],
        ["document"],
        ["title", "text"],
        ["instruction", "input", "output"],
        ["prompt", "completion"],
        ["question", "answer"],
    ]
    for group in fallback_groups:
        parts = [stringify(row.get(name)).strip() for name in group if row.get(name) is not None]
        parts = [p for p in parts if p]
        if parts:
            return "\n\n".join(parts)

    string_values = [stringify(v).strip() for v in row.values() if isinstance(v, str)]
    return "\n\n".join(v for v in string_values if v)


def cyrillic_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    cyr = sum(1 for ch in letters if "\u0400" <= ch <= "\u04ff")
    return cyr / max(1, len(letters))


class SequentialRussianPackedDataset(IterableDataset):
    def __init__(
        self,
        specs: List[Dict[str, Any]],
        tokenizer: Any,
        seq_len: int,
        seed: int,
        max_docs_per_dataset: int,
        shuffle_buffer: int,
        min_chars: int,
        min_cyrillic_ratio: float,
        append_eos: bool,
        trust_remote_code: bool,
        fail_on_dataset_error: bool,
    ) -> None:
        super().__init__()
        self.specs = specs
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.seed = seed
        self.max_docs_per_dataset = max_docs_per_dataset
        self.shuffle_buffer = shuffle_buffer
        self.min_chars = min_chars
        self.min_cyrillic_ratio = min_cyrillic_ratio
        self.append_eos = append_eos
        self.trust_remote_code = trust_remote_code
        self.fail_on_dataset_error = fail_on_dataset_error

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        buffer: List[int] = []
        eos_id = self.tokenizer.eos_token_id
        round_idx = 0

        while True:
            made_progress = False
            for dataset_idx, spec in enumerate(self.specs):
                path = spec["path"]
                name = spec.get("name")
                split = spec.get("split", "train")
                print(
                    f"[DATA] round={round_idx + 1} starting {dataset_idx + 1}/{len(self.specs)}: "
                    f"{path} {name or ''} split={split}",
                    flush=True,
                )

                try:
                    stream = load_dataset(
                        path,
                        name,
                        split=split,
                        streaming=True,
                        trust_remote_code=self.trust_remote_code,
                    )
                    if self.shuffle_buffer > 0:
                        stream = stream.shuffle(
                            buffer_size=self.shuffle_buffer,
                            seed=self.seed + dataset_idx + round_idx * 1009,
                        )
                except Exception as exc:
                    if self.fail_on_dataset_error:
                        raise
                    print(f"[WARN] skip dataset {path}: {exc}", flush=True)
                    continue

                docs_seen = 0
                samples_yielded = 0
                try:
                    for row in stream:
                        text = extract_text(row, spec)
                        if len(text) < self.min_chars:
                            continue
                        if self.min_cyrillic_ratio > 0 and cyrillic_ratio(text) < self.min_cyrillic_ratio:
                            continue

                        ids = self.tokenizer.encode(text, add_special_tokens=False)
                        if self.append_eos and eos_id is not None:
                            ids.append(eos_id)
                        if not ids:
                            continue

                        buffer.extend(ids)
                        while len(buffer) >= self.seq_len + 1:
                            chunk = buffer[: self.seq_len + 1]
                            del buffer[: self.seq_len]
                            input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                            labels = torch.tensor(chunk[1:], dtype=torch.long)
                            made_progress = True
                            samples_yielded += 1
                            yield {"input_ids": input_ids, "labels": labels}

                        docs_seen += 1
                        if self.max_docs_per_dataset > 0 and docs_seen >= self.max_docs_per_dataset:
                            break
                except Exception as exc:
                    if self.fail_on_dataset_error:
                        raise
                    print(f"[WARN] dataset interrupted {path}: {exc}", flush=True)

                print(f"[DATA] finished {path}; docs_seen={docs_seen}; samples={samples_yielded}", flush=True)

            round_idx += 1
            if self.max_docs_per_dataset <= 0 or not made_progress:
                break


def collate_batch(rows: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([r["input_ids"] for r in rows], dim=0),
        "labels": torch.stack([r["labels"] for r in rows], dim=0),
    }


def build_model(args: argparse.Namespace, vocab_size: int) -> EnergoAI:
    cfg = EnergoAIConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        latent_dim=args.latent_dim,
        state_dim=args.state_dim,
        coda_interval=args.coda_interval,
        prelude_positions=(-2, -1),
        max_seq_len=args.max_seq_len,
        mlp_intermediate_factor=args.mlp_intermediate_factor,
        efficient_intermediate_factor=args.efficient_intermediate_factor,
        use_flash_attention=not args.no_flash_attention,
        gradient_checkpointing=True,
    )
    model = EnergoAI(cfg)
    model.gradient_checkpointing = True
    return model


def get_model_dtype(args: argparse.Namespace) -> torch.dtype:
    if args.precision == "bf16":
        return torch.bfloat16
    if args.precision == "fp16":
        return torch.float16
    return torch.float32


def make_optimizer(args: argparse.Namespace, model: EnergoAI) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise SystemExit("Install bitsandbytes or use --optimizer adamw_torch") from exc
        return bnb.optim.AdamW8bit(
            params,
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )

    return torch.optim.AdamW(
        params,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        fused=torch.cuda.is_available(),
    )


def lr_for_step(args: argparse.Namespace, step: int) -> float:
    if step < args.warmup_steps:
        return args.learning_rate * max(1, step) / max(1, args.warmup_steps)
    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_learning_rate + (args.learning_rate - args.min_learning_rate) * cosine


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def save_checkpoint(
    output_dir: Path,
    step: int,
    model: EnergoAI,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    tokenizer: Any,
) -> None:
    ckpt_dir = output_dir / f"step_{step:08d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    raw_model = getattr(model, "_orig_mod", model)
    payload = {
        "step": step,
        "model": raw_model.state_dict(),
        "config": asdict(raw_model.config),
        "args": vars(args),
    }
    if args.save_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, ckpt_dir / "checkpoint.pt")
    tokenizer.save_pretrained(ckpt_dir / "tokenizer")

    checkpoints = sorted(output_dir.glob("step_*"))
    excess = len(checkpoints) - args.keep_last_checkpoints
    if args.keep_last_checkpoints > 0 and excess > 0:
        for old in checkpoints[:excess]:
            shutil.rmtree(old)


def load_checkpoint(path: str, model: EnergoAI) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    return int(ckpt.get("step", 0))


def main() -> None:
    args = parse_args()
    if args.print_default_manifest:
        print(json.dumps(DEFAULT_RU_DATASETS, ensure_ascii=False, indent=2))
        return

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"[GPU] {props.name}, VRAM={props.total_memory / 1024**3:.1f} GiB", flush=True)
        if props.total_memory < 31 * 1024**3:
            print("[WARN] This profile is intended for about 32GB VRAM.", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    vocab_size = len(tokenizer)
    print(f"[TOKENIZER] {args.tokenizer}, vocab_size={vocab_size}", flush=True)

    specs = load_dataset_manifest(args.dataset_manifest)
    with open(output_dir / "dataset_manifest.used.json", "w", encoding="utf-8") as f:
        json.dump(specs, f, ensure_ascii=False, indent=2)

    dataset = SequentialRussianPackedDataset(
        specs=specs,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        seed=args.seed,
        max_docs_per_dataset=args.max_docs_per_dataset,
        shuffle_buffer=args.shuffle_buffer,
        min_chars=args.min_chars,
        min_cyrillic_ratio=args.min_cyrillic_ratio,
        append_eos=args.append_eos,
        trust_remote_code=args.trust_remote_code,
        fail_on_dataset_error=args.fail_on_dataset_error,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        collate_fn=collate_batch,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args, vocab_size)
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model)
        print(f"[RESUME] loaded model from {args.resume}, step={start_step}", flush=True)

    model_dtype = get_model_dtype(args)
    model.to(device=device, dtype=model_dtype)
    model.train()

    if args.compile:
        model = torch.compile(model)

    optimizer = make_optimizer(args, model)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.precision == "fp16" and device.type == "cuda"))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] total_params={total_params:,}; trainable_params={trainable_params:,}", flush=True)
    print(
        f"[TRAIN] seq_len={args.seq_len}; micro_batch={args.micro_batch_size}; "
        f"grad_accum={args.grad_accum_steps}; precision={args.precision}; optimizer={args.optimizer}",
        flush=True,
    )

    data_iter = iter(loader)
    running_loss = 0.0
    running_tokens = 0
    last_log = time.time()

    pbar = tqdm(range(start_step + 1, args.max_steps + 1), initial=start_step, total=args.max_steps)
    for step in pbar:
        lr = lr_for_step(args, step)
        set_optimizer_lr(optimizer, lr)
        optimizer.zero_grad(set_to_none=True)

        step_loss = 0.0
        for _ in range(args.grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                print("[DATA] all datasets exhausted.", flush=True)
                save_checkpoint(output_dir, step - 1, model, optimizer, args, tokenizer)
                return

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=model_dtype, enabled=args.precision != "fp32"):
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                )
                loss = loss / args.grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            step_loss += float(loss.detach().cpu()) * args.grad_accum_steps
            running_tokens += int(input_ids.numel())

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        if args.max_grad_norm > 0:
            clip_grad_norm_(model.parameters(), args.max_grad_norm)

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        running_loss += step_loss

        if step % args.log_every == 0:
            now = time.time()
            elapsed = max(1e-6, now - last_log)
            tok_s = running_tokens / elapsed
            avg_loss = running_loss / args.log_every
            ppl = math.exp(min(20.0, avg_loss))
            mem = 0.0
            if device.type == "cuda":
                mem = torch.cuda.max_memory_allocated(device) / 1024**3
            print(
                f"[STEP {step}] loss={avg_loss:.4f} ppl={ppl:.2f} lr={lr:.3e} "
                f"tok/s={tok_s:.0f} max_vram={mem:.2f}GiB",
                flush=True,
            )
            pbar.set_description(f"loss={avg_loss:.3f}")
            running_loss = 0.0
            running_tokens = 0
            last_log = now

        if args.save_every > 0 and step % args.save_every == 0:
            save_checkpoint(output_dir, step, model, optimizer, args, tokenizer)

    save_checkpoint(output_dir, args.max_steps, model, optimizer, args, tokenizer)


if __name__ == "__main__":
    main()
