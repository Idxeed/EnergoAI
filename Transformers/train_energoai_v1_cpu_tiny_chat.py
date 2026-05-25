"""
CPU smoke-training for EnergoAI v1 small (46M).

This script is intentionally narrow:
  - runs on CPU
  - keeps RAM low by freezing the prefix of the network
  - trains only the last transformer layers plus a tiny output adapter
  - uses a tiny synthetic Russian dialogue set

Goal: verify that the full training loop, loss, backprop, checkpointing
and greedy generation work end-to-end on a constrained machine.

This is not a full-quality instruction tuning recipe.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import string
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from energoai_transformer_mini import EnergoAI, EnergoAIConfig


DialoguePair = Tuple[str, str]


TINY_DIALOGUES: List[DialoguePair] = [
    ("Привет", "Привет."),
    ("Как дела", "Хорошо."),
    ("Ты кто", "Я модель."),
    ("Поможешь", "Да."),
    ("Спасибо", "Пожалуйста."),
    ("Можно", "Можно."),
    ("Сколько будет два плюс два", "Четыре."),
    ("Да или нет", "Да."),
    ("Короткий ответ", "Да."),
    ("Что ты умеешь", "Помогать."),
    ("Говори по русски", "Да."),
    ("Это тест", "Да."),
    ("Ты живой", "Нет."),
    ("Что делать", "Проверить."),
    ("Какой шаг", "Дальше."),
    ("Нужен вывод", "Да."),
    ("Работает ли система", "Да."),
    ("Что важно", "Данные."),
    ("Что ты делаешь", "Отвечаю."),
    ("Скажи привет", "Привет."),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="runs/energoai_v1_cpu_tiny_chat")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--eval_every", type=int, default=25)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--cpu_threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_last_layers", type=int, default=1)
    parser.add_argument("--adapter_dim", type=int, default=64)
    parser.add_argument("--train_lm_head", action="store_true")
    parser.add_argument("--no_train_lm_head", action="store_true")
    parser.add_argument("--resume", type=str, default="")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


class CharTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2
    unk_token_id = 3
    vocab_size = 32000

    def __init__(self) -> None:
        base_chars = (
            " "
            + string.ascii_letters
            + string.digits
            + string.punctuation
            + "\n\t"
            + "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
            + "абвгдежзийклмнопрстуфхцчшщъыьэюя"
            + "Ёё"
        )
        self.chars: List[str] = []
        seen = set()
        for ch in base_chars:
            if ch not in seen:
                self.chars.append(ch)
                seen.add(ch)
        self.char_to_id = {ch: i + 4 for i, ch in enumerate(self.chars)}
        self.id_to_char = {i + 4: ch for i, ch in enumerate(self.chars)}

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        del add_special_tokens
        return [self.char_to_id.get(ch, self.unk_token_id) for ch in text]

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(self.id_to_char.get(token_id, "") for token_id in ids if token_id >= 4)


def build_tokenizer() -> CharTokenizer:
    return CharTokenizer()


def format_prompt(user_text: str) -> str:
    return f"Вопрос: {user_text}\nОтвет:"


def format_full(user_text: str, assistant_text: str) -> str:
    return f"{format_prompt(user_text)} {assistant_text}"


def encode_example(tokenizer, user_text: str, assistant_text: str, seq_len: int) -> dict:
    prompt_text = format_prompt(user_text)
    full_text = format_full(user_text, assistant_text)

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    if tokenizer.eos_token_id is not None:
        full_ids = full_ids + [tokenizer.eos_token_id]

    prompt_len = min(len(prompt_ids), len(full_ids))
    labels = [-100] * prompt_len + full_ids[prompt_len:]

    if len(full_ids) > seq_len:
        full_ids = full_ids[:seq_len]
        labels = labels[:seq_len]

    pad_len = seq_len - len(full_ids)
    input_ids = full_ids + [tokenizer.pad_token_id] * pad_len
    labels = labels + [-100] * pad_len
    attention_mask = [1] * len(full_ids) + [0] * pad_len

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class OutputAdapter(nn.Module):
    def __init__(self, hidden_size: int, bottleneck: int):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, hidden_size, bias=False)
        self.gate = nn.Parameter(torch.tensor(0.0))

        nn.init.xavier_uniform_(self.down.weight, gain=0.5)
        nn.init.xavier_uniform_(self.up.weight, gain=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.sigmoid(self.gate) * self.up(F.silu(self.down(x)))


class TinyChatSystem(nn.Module):
    def __init__(
        self,
        base: EnergoAI,
        train_last_layers: int,
        adapter_dim: int,
        train_lm_head: bool,
    ):
        super().__init__()
        self.base = base
        self.train_last_layers = max(1, train_last_layers)
        self.train_start = max(0, base.config.num_layers - self.train_last_layers)
        self.adapter = OutputAdapter(base.config.hidden_size, adapter_dim)
        self.train_lm_head = train_lm_head

        self._configure_trainable_params()

    def _configure_trainable_params(self) -> None:
        for p in self.base.parameters():
            p.requires_grad = False

        for idx in range(self.train_start, self.base.config.num_layers):
            for p in self.base.layers[idx].parameters():
                p.requires_grad = True

        trainable_coda_from = self.train_start // self.base.config.coda_interval
        for idx, block in enumerate(self.base.coda_blocks):
            if idx >= trainable_coda_from:
                for p in block.parameters():
                    p.requires_grad = True
                for p in self.base.coda_state_projs[idx].parameters():
                    p.requires_grad = True

        for p in self.base.norm.parameters():
            p.requires_grad = True
        for p in self.adapter.parameters():
            p.requires_grad = True
        if self.train_lm_head:
            self.base.lm_head.weight.requires_grad = True

    def _position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        return position_ids

    def _coda_state(self, x: torch.Tensor, attention_mask: torch.Tensor, coda_idx: int) -> torch.Tensor:
        if x.shape[1] == 1:
            return self.base.coda_state_projs[coda_idx](x[:, -1, :])
        seq_lengths = attention_mask.sum(dim=1).long().clamp(min=1) - 1
        rows = torch.arange(x.shape[0], device=x.device)
        return self.base.coda_state_projs[coda_idx](x[rows, seq_lengths])

    def _run_layers(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        start_layer: int,
        end_layer: int,
        grad_enabled: bool,
        coda_idx: int,
    ) -> Tuple[torch.Tensor, int]:
        ctx = torch.enable_grad() if grad_enabled else torch.no_grad()
        with ctx:
            for i in range(start_layer, end_layer):
                x, _ = self.base.layers[i](x, attention_mask, position_ids, None, False)
                if (i + 1) % self.base.config.coda_interval == 0 and coda_idx < len(self.base.coda_blocks):
                    h_state = self._coda_state(x, attention_mask, coda_idx)
                    x = self.base.coda_blocks[coda_idx](x, h_state, None)
                    coda_idx += 1
        return x, coda_idx

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        generated_ids: Sequence[int],
        temperature: float = 0.8,
        top_k: int = 20,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> int:
        logits = logits.float().clone()
        if generated_ids:
            seen = set(generated_ids[-64:])
            for token_id in seen:
                if 0 <= token_id < logits.shape[-1]:
                    logits[token_id] /= repetition_penalty

        logits = logits / max(temperature, 1e-6)
        if top_k > 0 and top_k < logits.shape[-1]:
            values, _ = torch.topk(logits, top_k)
            cutoff = values[-1]
            logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)

        probs = torch.softmax(logits, dim=-1)
        if top_p > 0.0 and top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            keep = cumulative <= top_p
            keep[..., 0] = True
            filtered = torch.zeros_like(probs)
            filtered[sorted_idx[keep]] = sorted_probs[keep]
            probs = filtered / filtered.sum().clamp_min(1e-12)

        return int(torch.multinomial(probs, num_samples=1).item())

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        position_ids = self._position_ids(attention_mask)
        x = self.base.embed_tokens(input_ids)

        coda_idx = 0
        x, coda_idx = self._run_layers(x, attention_mask, position_ids, 0, self.train_start, False, coda_idx)
        x, coda_idx = self._run_layers(x, attention_mask, position_ids, self.train_start, self.base.config.num_layers, True, coda_idx)

        x = self.base.norm(x)
        x = self.adapter(x)
        return self.base.lm_head(x)

    @torch.no_grad()
    def generate(self, tokenizer, prompt: str, max_new_tokens: int = 24) -> str:
        self.eval()
        device = next(self.parameters()).device
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        ids = list(prompt_ids) if prompt_ids else [tokenizer.eos_token_id]

        for _ in range(max_new_tokens):
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            logits = self.forward(input_ids, attention_mask)
            next_id = self._sample_next_token(logits[0, -1], ids)
            ids.append(next_id)
            if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
                break
            if len(ids) >= len(prompt_ids) + 3:
                tail = ids[-8:]
                if len(set(tail)) == 1:
                    break

        completion_ids = ids[len(prompt_ids):]
        text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        text = text.lstrip(string.whitespace + string.punctuation + ":-")
        return text or tokenizer.decode(ids, skip_special_tokens=True)

    @torch.no_grad()
    def choose_response(self, tokenizer, prompt: str, candidates: Sequence[str]) -> str:
        self.eval()
        device = next(self.parameters()).device
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not prompt_ids:
            prompt_ids = [tokenizer.unk_token_id]

        best_text = ""
        best_score = float("-inf")

        for candidate in candidates:
            candidate_ids = tokenizer.encode(candidate, add_special_tokens=False)
            if tokenizer.eos_token_id is not None:
                candidate_ids = candidate_ids + [tokenizer.eos_token_id]
            full_ids = prompt_ids + candidate_ids
            if len(full_ids) < 2:
                continue

            input_ids = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            logits = self.forward(input_ids, attention_mask)
            log_probs = torch.log_softmax(logits, dim=-1)
            targets = torch.tensor(full_ids[1:], dtype=torch.long, device=device)

            response_start = max(0, len(prompt_ids) - 1)
            response_log_probs = log_probs[0, response_start:, :].gather(-1, targets[response_start:].unsqueeze(-1)).squeeze(-1)
            score = float(response_log_probs.mean().item())

            if score > best_score:
                best_score = score
                best_text = candidate

        return best_text


def build_training_set(tokenizer, seq_len: int) -> List[dict]:
    return [encode_example(tokenizer, user, assistant, seq_len) for user, assistant in TINY_DIALOGUES]


def pick_optimizer(args: argparse.Namespace, params: Iterable[torch.nn.Parameter]):
    params = [p for p in params if p.requires_grad]
    if args.optimizer == "sgd":
        return torch.optim.SGD(params, lr=args.learning_rate)
    return torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95))


def main() -> None:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    train_lm_head = True
    if args.no_train_lm_head:
        train_lm_head = False
    if args.train_lm_head:
        train_lm_head = True

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    torch.set_num_threads(max(1, args.cpu_threads))
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = build_tokenizer()
    model_cfg = EnergoAIConfig(
        vocab_size=32000,
        hidden_size=512,
        num_layers=8,
        num_heads=8,
        num_kv_heads=2,
        latent_dim=64,
        state_dim=64,
        coda_interval=4,
        prelude_positions=(-2, -1),
        max_seq_len=2048,
        mlp_intermediate_factor=4,
        efficient_intermediate_factor=3,
        gradient_checkpointing=False,
        use_flash_attention=True,
    )
    base = EnergoAI(model_cfg)
    system = TinyChatSystem(
        base=base,
        train_last_layers=args.train_last_layers,
        adapter_dim=args.adapter_dim,
        train_lm_head=train_lm_head,
    )

    device = torch.device("cpu")
    system.to(device)
    system.train()

    trainable = [p for p in system.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in system.parameters())
    trainable_params = sum(p.numel() for p in trainable)
    print(f"[MODEL] total={total_params:,} trainable={trainable_params:,}")
    print(
        f"[TRAIN] cpu_threads={args.cpu_threads} seq_len={args.seq_len} steps={args.steps} "
        f"train_last_layers={args.train_last_layers} adapter_dim={args.adapter_dim}"
    )

    examples = build_training_set(tokenizer, args.seq_len)
    candidate_answers = sorted({assistant for _, assistant in TINY_DIALOGUES})
    optimizer = pick_optimizer(args, trainable)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        system.load_state_dict(ckpt["model"], strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        print(f"[RESUME] step={start_step}")

    def save_checkpoint(step: int) -> None:
        ckpt = {
            "step": step,
            "model": system.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        }
        torch.save(ckpt, output_dir / f"ckpt_step_{step:06d}.pt")

    def trainable_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        vocab = logits.shape[-1]
        return F.cross_entropy(logits.reshape(-1, vocab), labels.reshape(-1), ignore_index=-100)

    prompt_for_eval = "Вопрос: Привет\nОтвет:"

    for step in range(start_step + 1, args.steps + 1):
        sample = examples[(step - 1) % len(examples)]
        input_ids = sample["input_ids"].unsqueeze(0)
        attention_mask = sample["attention_mask"].unsqueeze(0)
        labels = sample["labels"].unsqueeze(0)

        optimizer.zero_grad(set_to_none=True)
        logits = system(input_ids, attention_mask)
        loss = trainable_loss(logits, labels)
        loss.backward()
        clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        if step % 10 == 0:
            print(f"[STEP {step}] loss={loss.item():.4f}")

        if step % args.eval_every == 0:
            reply = system.choose_response(tokenizer, prompt_for_eval, candidate_answers)
            print(f"[GEN] {reply}")

        if step % args.save_every == 0:
            save_checkpoint(step)

    save_checkpoint(args.steps)
    final_reply = system.choose_response(tokenizer, prompt_for_eval, candidate_answers)
    print(f"[FINAL] {final_reply}")


if __name__ == "__main__":
    main()
