"""
Быстрое обучение EnergoAI v5 на Tiny Shakespeare (~150 шагов)
Запуск: python train_v5_quick.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import requests
from collections import defaultdict

# Импорт модели v5 (файл должен называться energoai_ssm_v5.py)
from final_phase6.core.energoai_ssm_v5 import EnergoAIModel, SafeConfig


# ==========================================================
# 1. ПОДГОТОВКА ДАТАСЕТА (Tiny Shakespeare)
# ==========================================================
class TextDataset(Dataset):
    def __init__(self, text, seq_length=64):
        self.seq_length = seq_length
        chars = sorted(list(set(text)))
        self.stoi = {ch: i + 3 for i, ch in enumerate(chars)}
        self.stoi['<pad>'] = 0
        self.stoi['<bos>'] = 1
        self.stoi['<eos>'] = 2
        self.itos = {i: s for s, i in self.stoi.items()}
        self.data = [self.stoi.get(ch, 2) for ch in text]

    def __len__(self):
        return max(0, len(self.data) - self.seq_length)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx + self.seq_length], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + self.seq_length + 1], dtype=torch.long)
        return x, y


def download_tiny_shakespeare():
    print("📥 Скачиваем Tiny Shakespeare...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url)
    text = response.text
    print(f"✅ Загружено {len(text):,} символов")
    return text


# ==========================================================
# 2. TRAINING LOOP
# ==========================================================
def train():
    config = SafeConfig({
        "hidden_size": 256,
        "vocab_size": 257,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "intermediate_size": 512,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "attention_bias": False,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Device: {device}")
    print(f"📊 Config: hidden={config.hidden_size}, vocab={config.vocab_size}")

    text = download_tiny_shakespeare()
    dataset = TextDataset(text, seq_length=64)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

    print(f"📚 Dataset: {len(dataset)} примеров, {len(dataloader)} шагов/эпоха")

    model = EnergoAIModel(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model params: {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        eps=1e-8
    )

    warmup_steps = 50
    total_steps = 150
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, step / warmup_steps) if step < warmup_steps else 1.0
    )

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    metrics = defaultdict(list)

    print("\n" + "=" * 60)
    print("🚀 НАЧИНАЕМ ОБУЧЕНИЕ v5 (per-channel decay + input gate)")
    print("=" * 60)

    step = 0
    model.train()

    for epoch in range(3):
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
                avg_stop_prob = sum(s.get('stop_prob_mean', 0) for s in loop_stats) / len(loop_stats)
                avg_erase = sum(s.get('erase_gate_mean', 0) for s in loop_stats) / len(loop_stats)

                metrics['delta_norm'].append(avg_delta_norm)
                metrics['h_depth_norm'].append(avg_h_norm)
                metrics['h_depth_norm_fixed'].append(avg_h_fixed)
                metrics['decay'].append(avg_decay)
                metrics['stop_prob'].append(avg_stop_prob)
                metrics['erase'].append(avg_erase)

                # Собираем распределение Δ по каналам (из последнего цикла для примера)
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

            if step % 10 == 0 or step == total_steps - 1:
                delta_ch = metrics['Δ_per_channel'][-1]
                if delta_ch:
                    ch_info = f"Δ min={min(delta_ch):.5f} max={max(delta_ch):.3f}"
                else:
                    ch_info = ""
                print(f"Step {step:3d} | Loss: {loss.item():.4f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                      f"Δ_norm: {metrics['delta_norm'][-1]:.3f} | "
                      f"h_norm: {metrics['h_depth_norm'][-1]:.2f} | "
                      f"h_fixed: {metrics['h_depth_norm_fixed'][-1]:.2f} | "
                      f"stop: {metrics['stop_prob'][-1]:.3f} | "
                      f"erase: {metrics['erase'][-1]:.4f} | "
                      f"{ch_info}")

            step += 1

        if step >= total_steps:
            break

    print("\n" + "=" * 60)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("=" * 60)

    plot_training_metrics(metrics)

    # Тестовая генерация
    print("\n📝 Тестовая генерация текста:")
    model.eval()
    with torch.no_grad():
        start_text = text[:50]
        print(f"Start: {start_text}")
        input_ids = torch.tensor([[dataset.stoi.get(ch, 2) for ch in start_text]],
                                 dtype=torch.long).to(device)
        for _ in range(50):
            mask = (input_ids != 0).long()
            logits = model(input_ids, mask)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        generated = ''.join([dataset.itos.get(t.item(), '?') for t in input_ids[0]])
        print(f"Generated: {generated}")

    return model, metrics


def plot_training_metrics(metrics):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    # 1. Loss
    axes[0, 0].plot(metrics['step'], metrics['loss'], 'b-', linewidth=2)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True)

    # 2. Learning Rate
    axes[0, 1].plot(metrics['step'], metrics['lr'], 'g-', linewidth=2)
    axes[0, 1].set_title('Learning Rate')
    axes[0, 1].grid(True)

    # 3. Delta Norm
    axes[0, 2].plot(metrics['step'], metrics['delta_norm'], 'orange', linewidth=2)
    axes[0, 2].set_title('SSM Delta Norm')
    axes[0, 2].grid(True)

    # 4. Decay (теперь средний по каналам)
    axes[0, 3].plot(metrics['step'], metrics['decay'], 'red', linewidth=2)
    axes[0, 3].set_title('Depth Decay (mean)')
    axes[0, 3].set_ylim(0, 1.05)
    axes[0, 3].grid(True)

    # 5. H Depth Norm (raw)
    axes[1, 0].plot(metrics['step'], metrics['h_depth_norm'], 'purple', linewidth=2)
    axes[1, 0].set_title('H Depth Norm (raw)')
    axes[1, 0].grid(True)

    # 6. H Depth Norm (после RMSNorm)
    axes[1, 1].plot(metrics['step'], metrics['h_depth_norm_fixed'], 'cyan', linewidth=2)
    axes[1, 1].set_title('H Depth Norm (RMSNorm)')
    axes[1, 1].grid(True)

    # 7. Распределение Δ по каналам (последний шаг)
    if metrics['Δ_per_channel'] and metrics['Δ_per_channel'][-1]:
        last_delta = metrics['Δ_per_channel'][-1]
        axes[1, 2].bar(range(len(last_delta)), last_delta)
        axes[1, 2].set_title('Δ per channel (last step)')
        axes[1, 2].set_xlabel('Channel index')
        axes[1, 2].set_ylabel('Δ')
        axes[1, 2].grid(True)
    else:
        axes[1, 2].axis('off')

    # 8. Erase gate mean
    if 'erase' in metrics and any(metrics['erase']):
        axes[1, 3].plot(metrics['step'], metrics['erase'], 'magenta', linewidth=2)
        axes[1, 3].set_title('Erase Gate Mean')
        axes[1, 3].set_ylim(0, 1.05)
        axes[1, 3].grid(True)
    else:
        axes[1, 3].axis('off')

    plt.suptitle("EnergoAI v5 Training Metrics (150 steps)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("training_metrics_v5.png", dpi=150)
    print("📈 Графики сохранены в training_metrics_v5.png")
    plt.show()


if __name__ == "__main__":
    model, metrics = train()