"""
Быстрое обучение EnergoAI v6.5 на Tiny Shakespeare (~150 шагов)
Запуск: python train_v65.py

Требования: файл energoai_ssm_v6_5.py с классом EnergoAIModelV65 должен лежать
в той же директории (см. код модели в предыдущем сообщении).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import requests
from collections import defaultdict

from final_ssm_phase_beta_1.cores.energoai_ssm_v7 import EnergoAIModelV7, SafeConfig


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
    # v6.5 требует новых полей конфига
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
        # Новые параметры v6.5 (обязательны!)
        "num_prelude_layers": 2,
        "num_coda_layers": 2,
        "num_loops": 5,
        "state_dim": 128,
        "checkpoint_every": 500,
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Device: {device}")
    print(f"📊 Config: hidden={config.hidden_size}, loops={config.num_loops}, state={config.state_dim}")

    text = download_tiny_shakespeare()
    dataset = TextDataset(text, seq_length=64)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

    print(f"📚 Dataset: {len(dataset)} примеров, {len(dataloader)} шагов/эпоха")

    model = EnergoAIModelV7(config).to(device)
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
    print("🚀 НАЧИНАЕМ ОБУЧЕНИЕ v6.5 (Hard Constraint + Decoupled Δ + Forget Gate)")
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

            # === Сбор метрик v6.5 ===
            metrics['step'].append(step)
            metrics['loss'].append(loss.item())
            metrics['lr'].append(scheduler.get_last_lr()[0])

            loop_stats = diag.get('loop_stats', [])
            if loop_stats:
                # Новые ключи диагностики v6.5
                avg_delta_mean = sum(s.get('Δ_mean', 0) for s in loop_stats) / len(loop_stats)
                avg_a_bar = sum(s.get('A_bar_mean', 0) for s in loop_stats) / len(loop_stats)
                avg_g = sum(s.get('G_mean', 0) for s in loop_stats) / len(loop_stats)
                avg_h_raw = sum(s.get('h_norm_raw', 0) for s in loop_stats) / len(loop_stats)
                avg_h_clipped = sum(s.get('h_norm_clipped', 0) for s in loop_stats) / len(loop_stats)
                avg_h_fixed = sum(s.get('h_depth_norm_after_fix', 0) for s in loop_stats) / len(loop_stats)
                avg_decay = sum(s.get('decay', 0) for s in loop_stats) / len(loop_stats)
                avg_tau = sum(s.get('state_tau', 0) for s in loop_stats) / len(loop_stats)

                metrics['Δ_mean'].append(avg_delta_mean)
                metrics['A_bar_mean'].append(avg_a_bar)
                metrics['G_mean'].append(avg_g)
                metrics['h_norm_raw'].append(avg_h_raw)
                metrics['h_norm_clipped'].append(avg_h_clipped)
                metrics['h_depth_norm_fixed'].append(avg_h_fixed)
                metrics['decay'].append(avg_decay)
                metrics['state_tau'].append(avg_tau)

                # Распределение Δ по каналам (последний цикл последнего шага)
                last_loop = loop_stats[-1]
                if 'Δ_min' in last_loop and 'Δ_max' in last_loop:
                    metrics['Δ_min'].append(last_loop['Δ_min'])
                    metrics['Δ_max'].append(last_loop['Δ_max'])
                else:
                    metrics['Δ_min'].append(0.0)
                    metrics['Δ_max'].append(0.0)
            else:
                for k in ['Δ_mean', 'A_bar_mean', 'G_mean', 'h_norm_raw',
                          'h_norm_clipped', 'h_depth_norm_fixed', 'decay', 'state_tau']:
                    metrics[k].append(0.0)
                metrics['Δ_min'].append(0.0)
                metrics['Δ_max'].append(0.0)

            # === Логирование ===
            if step % 10 == 0 or step == total_steps - 1:
                print(f"Step {step:3d} | Loss: {loss.item():.5f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                      f"Δ_mean: {metrics['Δ_mean'][-1]:.5f} | "
                      f"A_bar: {metrics['A_bar_mean'][-1]:.5f} | "
                      f"G(forget): {metrics['G_mean'][-1]:.5f} | "
                      f"h_raw: {metrics['h_norm_raw'][-1]:.5f} | "
                      f"h_clip: {metrics['h_norm_clipped'][-1]:.5f} | "
                      f"h_fixed: {metrics['h_depth_norm_fixed'][-1]:.5f} | "
                      f"decay: {metrics['decay'][-1]:.5f}")

            step += 1

        if step >= total_steps:
            break

    print("\n" + "=" * 60)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("=" * 60)

    plot_training_metrics(metrics)

    # === Тестовая генерация ===
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

    # 3. Δ Mean (SSM step size)
    axes[0, 2].plot(metrics['step'], metrics['Δ_mean'], 'orange', linewidth=2)
    axes[0, 2].set_title('SSM Δ Mean')
    axes[0, 2].grid(True)

    # 4. Depth Decay (mean across loops)
    axes[0, 3].plot(metrics['step'], metrics['decay'], 'red', linewidth=2)
    axes[0, 3].set_title('Depth Decay (mean)')
    axes[0, 3].set_ylim(0, 1.05)
    axes[0, 3].grid(True)

    # 5. H Depth Norm (raw — ДО hard constraint)
    axes[1, 0].plot(metrics['step'], metrics['h_norm_raw'], 'purple', linewidth=2)
    axes[1, 0].set_title('H Depth Norm (raw, before clip)')
    axes[1, 0].grid(True)

    # 6. H Depth Norm (после RMSNorm + hard clip)
    axes[1, 1].plot(metrics['step'], metrics['h_depth_norm_fixed'], 'cyan', linewidth=2)
    axes[1, 1].set_title('H Depth Norm (after fix)')
    axes[1, 1].grid(True)

    # 7. Forget Gate G_mean
    axes[1, 2].plot(metrics['step'], metrics['G_mean'], 'brown', linewidth=2)
    axes[1, 2].set_title('Forget Gate G (mean)')
    axes[1, 2].set_ylim(0, 1.05)
    axes[1, 2].grid(True)

    # 8. A_bar Mean (effective decay factor)
    axes[1, 3].plot(metrics['step'], metrics['A_bar_mean'], 'teal', linewidth=2)
    axes[1, 3].set_title('A_bar Mean (effective decay)')
    axes[1, 3].set_ylim(0, 1.05)
    axes[1, 3].grid(True)

    plt.suptitle("EnergoAI v6.5 Training Metrics (150 steps)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("training_metrics_v65.png", dpi=150)
    print("📈 Графики сохранены в training_metrics_v65.png")
    plt.show()


if __name__ == "__main__":
    model, metrics = train()