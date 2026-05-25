"""
Обучение EnergoAI v5 масштаба ~0.2B параметров
Запуск: python train_v5_0.2B.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoTokenizer

from final_phase6.core.energoai_ssm_v5 import EnergoAIModel, SafeConfig

# ==========================================================
# 1. ДАТАСЕТ (TinyStories для теста, замени на русский позже)
# ==========================================================
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_length=256):
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        # Токенизируем все тексты и объединяем в один длинный список токенов
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
        self.tokens = all_tokens

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_length)

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.seq_length + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def load_tiny_stories(tokenizer, max_samples=5000):
    print("📥 Загружаем TinyStories...")
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    texts = []
    for i, sample in enumerate(dataset):
        if i >= max_samples:
            break
        texts.append(sample["text"])
    print(f"✅ Загружено {len(texts)} текстов")
    return TextDataset(texts, tokenizer, seq_length=256)


# ==========================================================
# 2. TRAINING LOOP
# ==========================================================
def train():
    config = SafeConfig({
        "hidden_size": 1024,
        "vocab_size": 50257,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "intermediate_size": 4096,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "attention_bias": False,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "num_prelude_layers": 4,
        "num_coda_layers": 3,
        "num_loops": 8,
        "state_dim": 256,
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Device: {device}")
    print(f"📊 Config: hidden={config.hidden_size}, vocab={config.vocab_size}")

    # Токенизатор GPT-2
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'pad_token': '<pad>'})  # pad_token_id = tokenizer.pad_token_id (будет 50257 или другой)

    dataset = load_tiny_stories(tokenizer, max_samples=20000)  # больше данных для большой модели
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)  # батч меньше из-за памяти

    print(f"📚 Dataset: {len(dataset)} примеров, {len(dataloader)} шагов/эпоха")

    model = EnergoAIModel(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model params: {total_params:,} (~{total_params/1e6:.1f}M)")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        eps=1e-8
    )

    warmup_steps = 200
    total_steps = 10000
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, step / warmup_steps) if step < warmup_steps else 1.0
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    metrics = defaultdict(list)

    print("\n" + "=" * 60)
    print("🚀 НАЧИНАЕМ ОБУЧЕНИЕ v5 0.2B (10000 шагов)")
    print("=" * 60)

    step = 0
    model.train()

    for epoch in range(20):  # много эпох, но остановимся по total_steps
        for batch_idx, (input_ids, targets) in enumerate(dataloader):
            if step >= total_steps:
                break

            input_ids = input_ids.to(device)
            targets = targets.to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long()

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits, diag = model.forward_with_diagnostics(input_ids, attention_mask)
                loss = F.cross_entropy(
                    logits.view(-1, config.vocab_size),
                    targets.view(-1),
                    ignore_index=tokenizer.pad_token_id
                )

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            # Сбор метрик
            metrics['step'].append(step)
            metrics['loss'].append(loss.item())
            metrics['lr'].append(scheduler.get_last_lr()[0])

            loop_stats = diag.get('loop_stats', [])
            if loop_stats:
                avg_delta_norm = sum(s.get('delta_norm', 0) for s in loop_stats) / len(loop_stats)
                avg_h_norm = sum(s.get('h_depth_norm', 0) for s in loop_stats) / len(loop_stats)
                avg_h_fixed = sum(s.get('h_depth_norm_after_fix', 0) for s in loop_stats) / len(loop_stats)
                avg_decay = sum(s.get('decay', 0) for s in loop_stats) / len(loop_stats)

                metrics['delta_norm'].append(avg_delta_norm)
                metrics['h_depth_norm'].append(avg_h_norm)
                metrics['h_depth_norm_fixed'].append(avg_h_fixed)
                metrics['decay'].append(avg_decay)

                last_loop = loop_stats[-1]
                if 'Δ_per_channel_mean' in last_loop:
                    metrics['Δ_per_channel'].append(last_loop['Δ_per_channel_mean'])
                else:
                    metrics['Δ_per_channel'].append([])
            else:
                metrics['delta_norm'].append(0.0)
                metrics['h_depth_norm'].append(0.0)
                metrics['h_depth_norm_fixed'].append(0.0)
                metrics['decay'].append(0.0)
                metrics['Δ_per_channel'].append([])

            if step % 50 == 0:
                delta_ch = metrics['Δ_per_channel'][-1]
                if delta_ch:
                    ch_info = f"Δ min={min(delta_ch):.4f} max={max(delta_ch):.3f}"
                else:
                    ch_info = ""
                print(f"Step {step:5d} | Loss: {loss.item():.4f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                      f"Δ_norm: {metrics['delta_norm'][-1]:.3f} | "
                      f"h_norm: {metrics['h_depth_norm'][-1]:.2f} | "
                      f"h_fixed: {metrics['h_depth_norm_fixed'][-1]:.2f} | "
                      f"{ch_info}")

            # Сохраняем чекпойнт каждые 1000 шагов
            if step % 1000 == 0 and step > 0:
                checkpoint_path = f"energoai_v5_0.2B_step{step}.pt"
                torch.save(model.state_dict(), checkpoint_path)
                print(f"💾 Чекпойнт сохранён: {checkpoint_path}")

            step += 1

        if step >= total_steps:
            break

    print("\n" + "=" * 60)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("=" * 60)

    # Сохраняем финальную модель
    torch.save(model.state_dict(), "energoai_v5_0.2B_final.pt")
    plot_training_metrics(metrics)

    # Тестовая генерация
    print("\n📝 Тестовая генерация текста:")
    model.eval()
    with torch.no_grad():
        prompt = "Once upon a time"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        for _ in range(100):
            mask = (input_ids != tokenizer.pad_token_id).long()
            logits, _ = model.forward_with_diagnostics(input_ids, mask)
            # Temperature sampling
            temperature = 0.8
            logits_temp = logits[:, -1, :] / temperature
            probs = F.softmax(logits_temp, dim=-1)
            next_token = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        generated = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print(f"Generated: {generated}")

    return model, metrics


def plot_training_metrics(metrics):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    axes[0, 0].plot(metrics['step'], metrics['loss'], 'b-', linewidth=2)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True)

    axes[0, 1].plot(metrics['step'], metrics['lr'], 'g-', linewidth=2)
    axes[0, 1].set_title('Learning Rate')
    axes[0, 1].grid(True)

    axes[0, 2].plot(metrics['step'], metrics['delta_norm'], 'orange', linewidth=2)
    axes[0, 2].set_title('SSM Delta Norm')
    axes[0, 2].grid(True)

    axes[0, 3].plot(metrics['step'], metrics['decay'], 'red', linewidth=2)
    axes[0, 3].set_title('Depth Decay (mean)')
    axes[0, 3].set_ylim(0, 1.05)
    axes[0, 3].grid(True)

    axes[1, 0].plot(metrics['step'], metrics['h_depth_norm'], 'purple', linewidth=2)
    axes[1, 0].set_title('H Depth Norm (raw)')
    axes[1, 0].grid(True)

    axes[1, 1].plot(metrics['step'], metrics['h_depth_norm_fixed'], 'cyan', linewidth=2)
    axes[1, 1].set_title('H Depth Norm (RMSNorm)')
    axes[1, 1].grid(True)

    if metrics['Δ_per_channel'] and metrics['Δ_per_channel'][-1]:
        last_delta = metrics['Δ_per_channel'][-1]
        axes[1, 2].bar(range(len(last_delta)), last_delta)
        axes[1, 2].set_title('Δ per channel (last step)')
        axes[1, 2].set_xlabel('Channel index')
        axes[1, 2].set_ylabel('Δ')
        axes[1, 2].grid(True)
    else:
        axes[1, 2].axis('off')

    axes[1, 3].axis('off')

    plt.suptitle("EnergoAI v5 0.2B Training Metrics", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("training_metrics_v5_0.2B.png", dpi=150)
    print("📈 Графики сохранены в training_metrics_v5_0.2B.png")
    plt.show()


if __name__ == "__main__":
    model, metrics = train()