"""
Финальное обучение EnergoAI 0.5B на диалогах (Alpaca + WikiText-103)
Запуск: python train_0.5B_final.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
from datasets import load_dataset
from transformers import AutoTokenizer

# Импорт твоей модели (лежит в том же каталоге)
from final_ssm_phase_beta_1.cores.energoai_ssm_v6 import EnergoAIModel, SafeConfig

warnings.filterwarnings('ignore')

# ==========================================================
# 1. КОНФИГУРАЦИЯ
# ==========================================================
def build_config(tokenizer):
    """Создаёт конфиг модели ~0.5B параметров."""
    return SafeConfig({
        "hidden_size": 1728,
        "vocab_size": len(tokenizer),
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "intermediate_size": 6912,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "attention_bias": False,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "num_prelude_layers": 8,
        "num_coda_layers": 4,
        "num_loops": 8,
        "state_dim": 512,
        "intermediate_multiplier": 6  # расширение MLP в Prelude v6
    })

# ==========================================================
# 2. ДАТАСЕТ
# ==========================================================
def load_and_prepare_dataset(tokenizer, seq_length=1024, max_samples=200_000):
    """Загружает Alpaca и WikiText-103, токенизирует и возвращает DataLoader."""
    print("📥 Загружаем Alpaca...")
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    def alpaca_format(example):
        prompt = f"Human: {example['instruction']}\n"
        if example.get('input'):
            prompt += f"{example['input']}\n"
        prompt += f"Assistant: {example.get('output', '')}"
        return {"text": prompt}
    alpaca = alpaca.map(alpaca_format, remove_columns=alpaca.column_names)

    print("📥 Загружаем WikiText-103...")
    wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    wiki = wiki.filter(lambda x: x["text"].strip() != "")
    wiki = wiki.map(lambda x: {"text": x["text"]}, remove_columns=wiki.column_names)

    # Объединяем тексты в один список
    texts = []
    for split in [alpaca, wiki]:
        for example in split:
            texts.append(example["text"])
            if len(texts) >= max_samples:
                break
        if len(texts) >= max_samples:
            break

    print(f"🔄 Токенизируем {len(texts)} текстов (длина блока {seq_length})...")
    all_chunks = []
    for text in texts:
        tokens = tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=seq_length
        )
        # Если последовательность короче, дополняем pad_token_id
        if len(tokens) < seq_length:
            tokens += [tokenizer.pad_token_id] * (seq_length - len(tokens))
        all_chunks.append(tokens)

    # Превращаем в тензор [N, seq_length]
    input_ids = torch.tensor(all_chunks, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(input_ids)
    print(f"✅ Получено {len(dataset)} блоков длиной {seq_length}")
    return dataset

# ==========================================================
# 3. ОБУЧЕНИЕ
# ==========================================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Устройство: {device}")

    # Токенизатор
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # используем eos как pad
    # Добавляем специальные токены для диалогов
    special_tokens = {"additional_special_tokens": ["Human:", "Assistant:"]}
    tokenizer.add_special_tokens(special_tokens)
    print(f"📚 Vocab size: {len(tokenizer)}")

    config = build_config(tokenizer)
    seq_length = 1024
    dataset = load_and_prepare_dataset(tokenizer, seq_length=seq_length, max_samples=200_000)
    dataloader = DataLoader(
        dataset,
        batch_size=2,           # маленький батч + градиентное аккумулирование
        shuffle=True,
        drop_last=True,
        collate_fn=lambda batch: torch.stack([item[0] for item in batch], dim=0)  # извлекаем тензор из кортежа, батч = [B, L]
    )

    model = EnergoAIModel(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Параметров: {total_params:,} (~{total_params/1e9:.2f}B)")

    # Параметры SSM-ядра получат свой learning rate (в 3 раза выше)
    # Разделяем параметры
    ssm_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'core_block' in name:
            ssm_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': 3e-4},
        {'params': ssm_params, 'lr': 1e-3}  # без warmup, сразу полный газ
    ], betas=(0.9, 0.95), weight_decay=0.01, eps=1e-8)

    # Шедулер: для основных параметров — warmup, для SSM — всегда 1.0 (LR не меняется)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[
            lambda step: min(1.0, step / warmup_steps) if step < warmup_steps else 1.0,  # other_params
            lambda step: 1.0  # ssm_params — без изменений
        ]
    )

    accumulation_steps = 8        # эффективный батч = 2 * 8 = 16
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    metrics = defaultdict(list)
    step = 0
    model.train()
    print(f"\n{'='*60}")
    print(f"🚀 ФИНАЛЬНОЕ ОБУЧЕНИЕ 0.5B (20000 шагов)")
    print(f"{'='*60}")

    # Бесконечный цикл по даталоадеру (с повторением эпох)
    while step < total_steps:
        for batch in dataloader:
            if step >= total_steps:
                break

            # batch: [B, L]
            input_ids = batch[:, :-1].to(device)
            targets = batch[:, 1:].to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long()

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits, diag = model.forward_with_diagnostics(input_ids, attention_mask)
                loss = F.cross_entropy(
                    logits.view(-1, config.vocab_size),
                    targets.reshape(-1),
                    ignore_index=tokenizer.pad_token_id
                )
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            # Сбор метрик
            metrics['step'].append(step)
            metrics['loss'].append(loss.item() * accumulation_steps)
            metrics['lr'].append(scheduler.get_last_lr()[0])

            loop_stats = diag.get('loop_stats', [])
            if loop_stats:
                avg_delta_norm = sum(s.get('delta_norm', 0) for s in loop_stats) / len(loop_stats)
                avg_h_norm = sum(s.get('h_depth_norm', 0) for s in loop_stats) / len(loop_stats)
                avg_h_fixed = sum(s.get('h_depth_norm_after_fix', 0) for s in loop_stats) / len(loop_stats)
                avg_decay = sum(s.get('decay', 0) for s in loop_stats) / len(loop_stats)
                avg_stop = sum(s.get('stop_prob_mean', 0) for s in loop_stats) / len(loop_stats)
                avg_erase = sum(s.get('erase_gate_mean', 0) for s in loop_stats) / len(loop_stats)

                metrics['delta_norm'].append(avg_delta_norm)
                metrics['h_depth_norm'].append(avg_h_norm)
                metrics['h_depth_norm_fixed'].append(avg_h_fixed)
                metrics['decay'].append(avg_decay)
                metrics['stop_prob'].append(avg_stop)
                metrics['erase'].append(avg_erase)

                last_loop = loop_stats[-1]
                if 'Δ_per_channel_mean' in last_loop:
                    metrics['Δ_per_channel'].append(last_loop['Δ_per_channel_mean'])
                else:
                    metrics['Δ_per_channel'].append([])

            if step % 1 == 0:
                delta_ch = metrics['Δ_per_channel'][-1] if metrics['Δ_per_channel'] else []
                ch_info = f"Δ min={min(delta_ch):.4f} max={max(delta_ch):.3f}" if delta_ch else ""
                print(f"Step {step:5d} | Loss: {metrics['loss'][-1]:.4f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                      f"Δ_norm: {metrics['delta_norm'][-1]:.5f} | "
                      f"h_norm: {metrics['h_depth_norm'][-1]:.2f} | "
                      f"h_fixed: {metrics['h_depth_norm_fixed'][-1]:.2f} | "
                      f"stop: {metrics['stop_prob'][-1]:.3f} | "
                      f"erase: {metrics['erase'][-1]:.4f} | "
                      f"{ch_info}")

            if step % 2000 == 0 and step > 0:
                ckpt_path = f"energoai_0.5B_step{step}.pt"
                torch.save(model.state_dict(), ckpt_path)
                print(f"💾 Чекпойнт сохранён: {ckpt_path}")

            step += 1

    # Финальное сохранение
    torch.save(model.state_dict(), "energoai_0.5B_final.pt")
    print("\n✅ Обучение завершено!")

    # Быстрый тест генерации
    print("\n📝 Пробная генерация диалога:")
    model.eval()
    with torch.no_grad():
        prompt = "Human: What is the capital of France?\nAssistant:"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        for _ in range(50):
            mask = (input_ids != tokenizer.pad_token_id).long()
            logits, _ = model.forward_with_diagnostics(input_ids, mask)
            temperature = 0.7
            logits_temp = logits[:, -1, :] / temperature
            probs = F.softmax(logits_temp, dim=-1)
            next_token = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        generated = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print(generated)

    # Простейший график (сохраним)
    plot_metrics(metrics)

def plot_metrics(m):
    plt.figure(figsize=(10, 6))
    plt.plot(m['step'], m['loss'], label='Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig("loss_0.5B.png")
    plt.close()

if __name__ == "__main__":
    train()