# train_v711_tiny_shakespeare.py
"""
Быстрое обучение EnergoAI v7.1.1 на Tiny Shakespeare (~150 шагов)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import requests
from collections import defaultdict

from final_ssm_phase_beta_1.cores.energoai_ssm_v7_1_1 import EnergoAIModelV711, SafeConfig


# ==========================================================
# 1. ДАТАСЕТ
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
        "num_prelude_layers": 2,
        "num_coda_layers": 2,
        "num_loops": 4,
        "state_dim": 128,
        "checkpoint_every": 500,
        "transformer_heads": 4,
        "transformer_kv_heads": 4,
        "transformer_latent": 128,
        "curriculum_phase": 0,
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Device: {device}")

    text = download_tiny_shakespeare()
    dataset = TextDataset(text, seq_length=64)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

    print(f"📚 Dataset: {len(dataset)} примеров")

    model = EnergoAIModelV711(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model params: {total_params:,} ({total_params / 1e6:.2f}M)")

    # GPU оптимизации
    batch_size = 64
    seq_length = 128
    total_steps = 1000
    curriculum_switch_step = 200
    warmup_steps = 100

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        eps=1e-8
    )

    # Cosine annealing с warmup
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: (step / warmup_steps) if step < warmup_steps
        else 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
    )

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    metrics = defaultdict(list)

    print("\n" + "=" * 70)
    print("🚀 ENERGOAI v7.1.1")
    print(f"   Фаза 0 (0-{curriculum_switch_step}): SSM bootcamp")
    print(f"   Фаза 1 ({curriculum_switch_step}+): SSM + Transformer")
    print("=" * 70)

    step = 0
    model.train()

    for epoch in range(3):
        for batch_idx, (input_ids, targets) in enumerate(dataloader):
            if step >= total_steps:
                break

            if step == curriculum_switch_step:
                model.set_curriculum_phase(1)
                print(f"\n🔓 Step {step}: Transformer UNLOCKED")

            input_ids = input_ids.to(device)
            targets = targets.to(device)
            attention_mask = (input_ids != 0).long()

            # === FORWARD + LOSS ===
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits, diag, loss_aux_tensor = model.forward_with_diagnostics(input_ids, attention_mask)
                    loss_lm = F.cross_entropy(
                        logits.view(-1, config.vocab_size),
                        targets.view(-1),
                        ignore_index=0
                    )

                    # Curriculum: фаза 0 = high aux (градиенты в SSM!)
                    if step < curriculum_switch_step:
                        loss = loss_lm + 3.0 * loss_aux_tensor
                    else:
                        loss = loss_lm + 0.3 * loss_aux_tensor

                    loss_aux = loss_aux_tensor.item()

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, diag, loss_aux_tensor = model.forward_with_diagnostics(input_ids, attention_mask)
                loss_lm = F.cross_entropy(
                    logits.view(-1, config.vocab_size),
                    targets.view(-1),
                    ignore_index=0
                )

                if step < curriculum_switch_step:
                    loss = loss_lm + 3.0 * loss_aux_tensor
                else:
                    loss = loss_lm + 0.3 * loss_aux_tensor

                loss_aux = loss_aux_tensor.item()
                loss.backward()
                for name, p in model.named_parameters():
                    if p.grad is not None and 'ssm' in name:
                        print(f"{name}: grad_norm={p.grad.norm():.6f}")
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()

            # === МЕТРИКИ ===
            metrics['step'].append(step)
            metrics['loss'].append(loss.item())
            metrics['loss_lm'].append(loss_lm.item())
            metrics['loss_aux'].append(loss_aux)
            metrics['lr'].append(scheduler.get_last_lr()[0])
            metrics['transformer_gate'].append(diag.get('transformer_gate', 0.0))

            # SSM
            metrics['ssm_h_norm'].append(diag.get('ssm_h_norm', 0.0))
            metrics['ssm_delta_mean'].append(diag.get('ssm_delta_mean', 0.0))
            metrics['ssm_delta_std'].append(diag.get('ssm_delta_std', 0.0))
            metrics['ssm_delta_min'].append(diag.get('ssm_delta_min', 0.0))
            metrics['ssm_delta_max'].append(diag.get('ssm_delta_max', 0.0))
            metrics['ssm_A_bar'].append(diag.get('ssm_A_bar', 0.0))
            metrics['ssm_A_bar_min'].append(diag.get('ssm_A_bar_min', 0.0))
            metrics['ssm_A_bar_max'].append(diag.get('ssm_A_bar_max', 0.0))
            metrics['ssm_G_mean'].append(diag.get('ssm_G_mean', 0.0))
            metrics['ssm_G_min'].append(diag.get('ssm_G_min', 0.0))
            metrics['ssm_G_max'].append(diag.get('ssm_G_max', 0.0))
            metrics['ssm_W_in_rank'].append(diag.get('ssm_W_in_rank', 0))
            metrics['ssm_active_channels'].append(diag.get('ssm_active_channels', 0))
            metrics['ssm_delta_scale'].append(diag.get('ssm_delta_scale', 0.0))
            metrics['ssm_input_gain'].append(diag.get('ssm_input_gain', 0.0))

            # Нормы
            metrics['prelude_norm'].append(diag.get('prelude_norm', 0.0))
            metrics['transformer_norm'].append(diag.get('transformer_norm', 0.0))
            metrics['coda_norm'].append(diag.get('coda_norm', 0.0))
            metrics['h_depth_norm'].append(diag.get('h_depth_norm', 0.0))

            # === ЛОГ ===
            if step % 10 == 0 or step == total_steps - 1:
                phase_str = "SSM" if model.curriculum_phase.item() == 0 else "SSM+TF"
                print(f"Step {step:3d} [{phase_str}] | "
                      f"Loss: {loss.item():.5f} (LM:{loss_lm:.5f}+Aux:{loss_aux:.5f}) | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                      f"Gate: {diag.get('transformer_gate', 0.0):.3f} | "
                      f"SSM h: {diag.get('ssm_h_norm', 0.0):.3f} | "
                      f"Δ: {diag.get('ssm_delta_mean', 0.0):.3f} [{diag.get('ssm_delta_min', 0.0):.3f}-{diag.get('ssm_delta_max', 0.0):.3f}] | "
                      f"Δstd: {diag.get('ssm_delta_std', 0.0):.3f} | "
                      f"A_bar: {diag.get('ssm_A_bar', 0.0):.4f} [{diag.get('ssm_A_bar_min', 0.0):.4f}-{diag.get('ssm_A_bar_max', 0.0):.4f}] | "
                      f"G: {diag.get('ssm_G_mean', 0.0):.4f} [{diag.get('ssm_G_min', 0.0):.4f}-{diag.get('ssm_G_max', 0.0):.4f}] | "
                      f"Rank: {diag.get('ssm_W_in_rank', 0)}/{config.state_dim} | "
                      f"Active: {diag.get('ssm_active_channels', 0)}/{config.state_dim} | "
                      f"δscale: {diag.get('ssm_delta_scale', 0.0):.3f} | "
                      f"Gain: {diag.get('ssm_input_gain', 0.0):.3f} | "
                      f"Aux: {loss_aux:.4f}")

            step += 1

        if step >= total_steps:
            break

    print("\n" + "=" * 70)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("=" * 70)

    plot_training_metrics(metrics)

    # Генерация
    print("\n📝 Тестовая генерация:")
    model.eval()
    with torch.no_grad():
        start_text = text[:50]
        print(f"Start: '{start_text}'")
        input_ids = torch.tensor([[dataset.stoi.get(ch, 2) for ch in start_text]],
                                 dtype=torch.long).to(device)
        for _ in range(100):
            mask = (input_ids != 0).long()
            logits = model(input_ids, mask)
            probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
            next_token = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        generated = ''.join([dataset.itos.get(t.item(), '?') for t in input_ids[0]])
        print(f"Generated: '{generated}'")

    return model, metrics


# ==========================================================
# 3. ГРАФИКИ
# ==========================================================
def plot_training_metrics(metrics):
    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    steps = metrics['step']

    # Row 1
    axes[0, 0].plot(steps, metrics['loss'], 'b-', label='Total')
    axes[0, 0].plot(steps, metrics['loss_lm'], 'g--', alpha=0.7, label='LM')
    axes[0, 0].plot(steps, metrics['loss_aux'], 'r:', alpha=0.7, label='Aux')
    axes[0, 0].axvline(x=50, color='orange', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True)

    axes[0, 1].plot(steps, metrics['lr'], 'g-')
    axes[0, 1].set_title('LR')
    axes[0, 1].grid(True)

    axes[0, 2].plot(steps, metrics['transformer_gate'], 'purple')
    axes[0, 2].axvline(x=50, color='orange', linestyle='--', alpha=0.5)
    axes[0, 2].set_title('Transformer Gate')
    axes[0, 2].set_ylim(-0.05, 1.05)
    axes[0, 2].grid(True)

    axes[0, 3].plot(steps, metrics['ssm_delta_scale'], 'red')
    axes[0, 3].set_title('Delta Scale')
    axes[0, 3].grid(True)

    # Row 2
    axes[1, 0].plot(steps, metrics['ssm_h_norm'], 'red')
    axes[1, 0].set_title('SSM h_norm')
    axes[1, 0].grid(True)

    axes[1, 1].plot(steps, metrics['ssm_delta_mean'], 'orange')
    axes[1, 1].fill_between(steps, metrics['ssm_delta_min'], metrics['ssm_delta_max'], alpha=0.3)
    axes[1, 1].set_title('Δ Spectrum')
    axes[1, 1].grid(True)

    axes[1, 2].plot(steps, metrics['ssm_A_bar'], 'teal')
    axes[1, 2].fill_between(steps, metrics['ssm_A_bar_min'], metrics['ssm_A_bar_max'], alpha=0.3)
    axes[1, 2].set_title('A_bar')
    axes[1, 2].set_ylim(0, 1.05)
    axes[1, 2].grid(True)

    axes[1, 3].plot(steps, metrics['ssm_delta_std'], 'orange')
    axes[1, 3].set_title('Δ Std (Diversity)')
    axes[1, 3].grid(True)

    # Row 3
    axes[2, 0].plot(steps, metrics['prelude_norm'], 'blue', label='Prelude')
    axes[2, 0].plot(steps, metrics['transformer_norm'], 'purple', label='Transformer', alpha=0.7)
    axes[2, 0].plot(steps, metrics['coda_norm'], 'green', label='Coda', alpha=0.7)
    axes[2, 0].axvline(x=50, color='orange', linestyle='--', alpha=0.5)
    axes[2, 0].set_title('Layer Norms')
    axes[2, 0].legend(fontsize=8)
    axes[2, 0].grid(True)

    axes[2, 1].plot(steps, metrics['ssm_G_mean'], 'purple')
    axes[2, 1].fill_between(steps, metrics['ssm_G_min'], metrics['ssm_G_max'], alpha=0.3)
    axes[2, 1].set_title('Forget Gate G')
    axes[2, 1].set_ylim(0, 1.05)
    axes[2, 1].grid(True)

    axes[2, 2].plot(steps, metrics['ssm_active_channels'], 'green')
    axes[2, 2].axhline(y=128, color='gray', linestyle='--', alpha=0.5)
    axes[2, 2].set_title('Active Channels')
    axes[2, 2].grid(True)

    axes[2, 3].plot(steps, metrics['ssm_input_gain'], 'brown')
    axes[2, 3].set_title('Input Gain')
    axes[2, 3].grid(True)

    plt.suptitle("EnergoAI v7.1.1 Training Metrics", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("training_metrics_v711.png", dpi=150)
    print("📈 Графики сохранены")
    plt.show()


if __name__ == "__main__":
    model, metrics = train()