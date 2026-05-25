"""
Обучение EnergoAI v6.5 120M с НУЛЯ (собственный BPE, без Llama)
Запуск: python train_v65_120m_from_scratch.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import numpy as np
from collections import defaultdict
import json, os, time

from final_ssm_phase_beta_1.cores.energoai_ssm_v6_5 import EnergoAIModelV65, SafeConfig


# ==========================================
# 1. CONFIG 120M
# ==========================================
def get_config_120m(vocab_size=32000):
    return SafeConfig({
        "hidden_size": 768,
        "vocab_size": vocab_size,
        "num_attention_heads": 12,
        "num_key_value_heads": 4,
        "intermediate_size": 2048,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "attention_bias": False,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "num_prelude_layers": 4,
        "num_coda_layers": 4,
        "num_loops": 8,
        "state_dim": 256,
        "checkpoint_every": 1024,
    })


# ==========================================
# 2. TOKENIZER С НУЛЯ (BPE)
# ==========================================
def train_bpe_tokenizer(text_iterator, vocab_size=32000, save_path="energoai_tokenizer.json"):
    if os.path.exists(save_path):
        print(f"✅ Загружаем tokenizer: {save_path}")
        return Tokenizer.from_file(save_path)

    print(f"🔧 Обучаем BPE (vocab={vocab_size})...")

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train_from_iterator(text_iterator, trainer=trainer)
    tokenizer.save(save_path)

    print(f"💾 Tokenizer сохранён: {save_path}")
    return tokenizer


class StreamingTextIterator:
    def __init__(self, dataset_stream, max_samples=100_000):
        self.stream = dataset_stream
        self.max_samples = max_samples
        self.count = 0

    def __iter__(self):
        for sample in self.stream:
            if self.count >= self.max_samples:
                break
            text = sample.get("text", "")
            if len(text) > 100:
                yield text
                self.count += 1


# ==========================================
# 3. DATASET (Fineweb-Edu)
# ==========================================
class FinewebDataset(IterableDataset):
    def __init__(self, tokenizer, seq_length=1024, total_tokens=1_300_000_000):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.total_tokens = total_tokens

        # Fineweb-edu: 1.3B токенов, открытый, стабильный
        self.stream = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            "sample-10BT",  # 10B токенов, берём часть
            split="train",
            streaming=True
        )

        self.buffer = []
        self.buffer_tokens = 0

    def encode(self, text):
        encoding = self.tokenizer.encode(text)
        ids = [1] + encoding.ids + [2]
        return ids

    def __iter__(self):
        for sample in self.stream:
            text = sample.get("text", "")
            if len(text) < 100:
                continue

            tokens = self.encode(text)
            self.buffer.extend(tokens)
            self.buffer_tokens += len(tokens)

            while len(self.buffer) >= self.seq_length + 1:
                x = torch.tensor(self.buffer[:self.seq_length], dtype=torch.long)
                y = torch.tensor(self.buffer[1:self.seq_length+1], dtype=torch.long)
                self.buffer = self.buffer[self.seq_length:]
                yield x, y

            if self.buffer_tokens >= self.total_tokens:
                break


# ==========================================
# 4. TRAINING LOOP (single GPU, no distributed)
# ==========================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")

    # === Шаг 1: Обучаем tokenizer ===
    print("📚 Этап 1: Обучение tokenizer...")

    try:
        temp_stream = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            "sample-10BT",
            split="train",
            streaming=True
        )
    except Exception as e:
        print(f"⚠️ Fineweb недоступен: {e}")
        print("Создаём dummy tokenizer для теста...")
        # Fallback: обучаем на локальном тексте
        dummy_text = ["Hello world " * 1000] * 1000
        tokenizer = train_bpe_tokenizer(iter(dummy_text), vocab_size=32000)
        vocab_size = tokenizer.get_vocab_size()
        print(f"✅ Dummy vocab size: {vocab_size}")
        # Используем dummy dataset
        from torch.utils.data import TensorDataset
        dummy_ids = torch.randint(0, vocab_size, (1000, 1024))
        dataloader = DataLoader(TensorDataset(dummy_ids, dummy_ids), batch_size=8, shuffle=True)
        config = get_config_120m(vocab_size=vocab_size)
        model = EnergoAIModelV65(config).to(device)
        print("⚠️ Режим тестирования (dummy data)")
        return  # Выходим для теста

    text_iterator = StreamingTextIterator(temp_stream, max_samples=100_000)
    tokenizer = train_bpe_tokenizer(text_iterator, vocab_size=32000, save_path="energoai_tokenizer.json")
    vocab_size = tokenizer.get_vocab_size()
    print(f"✅ Vocab size: {vocab_size}")

    # === Шаг 2: Модель ===
    config = get_config_120m(vocab_size=vocab_size)
    model = EnergoAIModelV65(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model: {total_params/1e6:.1f}M params")

    # === Шаг 3: Dataset ===
    dataset = FinewebDataset(tokenizer, seq_length=1024, total_tokens=1_300_000_000)
    dataloader = DataLoader(dataset, batch_size=8, num_workers=2, pin_memory=True)

    # === Шаг 4: Optimizer ===
    try:
        from muon import Muon
        muon_params = [p for n, p in model.named_parameters()
                      if p.ndim >= 2 and "embed" not in n and "norm" not in n]
        adamw_params = [p for n, p in model.named_parameters()
                       if not (p.ndim >= 2 and "embed" not in n and "norm" not in n)]
        optimizer = Muon(muon_params, lr=0.02, momentum=0.95,
                        adamw_params=adamw_params, adamw_lr=3e-4, adamw_wd=0.01)
        print("✅ Muon")
    except ImportError:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=3e-4, betas=(0.9, 0.95),
            weight_decay=0.01, eps=1e-8
        )
        print("⚠️ AdamW")

    total_steps = 20_000
    warmup_steps = 2_000

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.1 + 0.9 * (0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    os.makedirs("checkpoints_v65_120m", exist_ok=True)

    # === Шаг 5: Training ===
    metrics = defaultdict(list)
    step = 0
    model.train()
    start_time = time.time()

    for batch_idx, (input_ids, targets) in enumerate(dataloader):
        if step >= total_steps:
            break

        input_ids = input_ids.to(device)
        targets = targets.to(device)
        attention_mask = (input_ids != 0).long()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits, diag = model.forward_with_diagnostics(input_ids, attention_mask)
                loss = F.cross_entropy(
                    logits.view(-1, config.vocab_size),
                    targets.view(-1),
                    ignore_index=0
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, diag = model.forward_with_diagnostics(input_ids, attention_mask)
            loss = F.cross_entropy(
                logits.view(-1, config.vocab_size),
                targets.view(-1),
                ignore_index=0
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        optimizer.zero_grad()
        scheduler.step()

        metrics['step'].append(step)
        metrics['loss'].append(loss.item())
        metrics['lr'].append(scheduler.get_last_lr()[0])

        if step % 1 == 0:
            loop_stats = diag.get('loop_stats', [])

            # === SSM Core метрики ===
            if loop_stats:
                last_loop = loop_stats[-1]

                # State
                avg_h_raw = np.mean([s.get('h_norm_raw', 0) for s in loop_stats])
                avg_h_clipped = np.mean([s.get('h_norm_clipped', 0) for s in loop_stats])
                avg_h_fixed = np.mean([s.get('h_depth_norm_after_fix', 0) for s in loop_stats])

                # Delta
                avg_delta = np.mean([s.get('Δ_mean', 0) for s in loop_stats])
                delta_min = last_loop.get('Δ_min', 0)
                delta_max = last_loop.get('Δ_max', 0)

                # Dynamics
                avg_a_bar = np.mean([s.get('A_bar_mean', 0) for s in loop_stats])
                avg_g = np.mean([s.get('G_mean', 0) for s in loop_stats])
                avg_decay = np.mean([s.get('decay', 0) for s in loop_stats])
                avg_tau = np.mean([s.get('state_tau', 0) for s in loop_stats])

                # Gates / scales
                avg_input_gain = np.mean([s.get('input_gain', 0) for s in loop_stats])
                avg_delta_scale = np.mean([s.get('delta_scale', 0) for s in loop_stats])
            else:
                avg_h_raw = avg_h_clipped = avg_h_fixed = 0
                avg_delta = delta_min = delta_max = 0
                avg_a_bar = avg_g = avg_decay = avg_tau = 0
                avg_input_gain = avg_delta_scale = 0

            # === Глобальные метрики ===
            elapsed = time.time() - start_time
            tok_per_sec = (step * 8192) / elapsed if elapsed > 0 else 0

            # Скользящее среднее loss
            if step == 0:
                ema_loss = loss.item()
            else:
                ema_alpha = 0.95
                ema_loss = ema_alpha * metrics['ema_loss'][-1] + (1 - ema_alpha) * loss.item()

            # === Сохраняем в metrics ===
            metrics['step'].append(step)
            metrics['loss'].append(loss.item())
            metrics['ema_loss'].append(ema_loss)
            metrics['lr'].append(scheduler.get_last_lr()[0])
            metrics['tok_per_sec'].append(tok_per_sec)
            metrics['h_raw'].append(avg_h_raw)
            metrics['h_clipped'].append(avg_h_clipped)
            metrics['h_fixed'].append(avg_h_fixed)
            metrics['delta_mean'].append(avg_delta)
            metrics['delta_min'].append(delta_min)
            metrics['delta_max'].append(delta_max)
            metrics['a_bar'].append(avg_a_bar)
            metrics['g_mean'].append(avg_g)
            metrics['decay'].append(avg_decay)
            metrics['state_tau'].append(avg_tau)
            metrics['input_gain'].append(avg_input_gain)
            metrics['delta_scale'].append(avg_delta_scale)

            # === Вывод ===
            print(f"Step {step:5d} | "
                  f"Loss: {loss.item():.4f} | "
                  f"EMA: {ema_loss:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                  f"h_raw: {avg_h_raw:.2f} | "
                  f"h_clip: {avg_h_clipped:.2f} | "
                  f"h_fix: {avg_h_fixed:.2f} | "
                  f"Δ: {avg_delta:.3f} [{delta_min:.3f}-{delta_max:.3f}] | "
                  f"A_bar: {avg_a_bar:.4f} | "
                  f"G: {avg_g:.4f} | "
                  f"decay: {avg_decay:.3f} | "
                  f"gain: {avg_input_gain:.3f} | "
                  f"Tok/s: {tok_per_sec:.0f}")

        if step > 0 and step % 5_000 == 0:
            ckpt_path = f"checkpoints_v65_120m/step_{step}.pt"
            torch.save({
                'step': step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'metrics': dict(metrics),
                'tokenizer_path': "energoai_tokenizer.json",
            }, ckpt_path)
            print(f"💾 Checkpoint: {ckpt_path}")

        step += 1

    final_path = "checkpoints_v65_120m/final.pt"
    torch.save({
        'step': step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'metrics': dict(metrics),
        'tokenizer_path': "energoai_tokenizer.json",
    }, final_path)

    with open("checkpoints_v65_120m/metrics.json", "w") as f:
        json.dump(dict(metrics), f)

    print(f"\n✅ Готово: {step} шагов, {step*8192/1e9:.2f}B токенов")
    print(f"Final loss: {metrics['loss'][-1]:.4f}")


if __name__ == "__main__":
    train()