# test_v3.py
"""
Smoke-тест EnergoAI SSM v3 с визуализацией.
Проверяет, что модель не взрывается, нормы стабильны.
Запуск: python test_v3.py
"""

import torch
import matplotlib.pyplot as plt

# Импорт из файлов проекта (должны лежать рядом)
from final_phase3.core.energoai_ssm_v3 import EnergoAIModel, SafeConfig


# ------------------- Конфигурация -------------------
def get_small_config():
    return SafeConfig({
        "hidden_size": 512,
        "vocab_size": 10000,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "intermediate_size": 1024,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "attention_bias": False,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    })

# ------------------- Тест -------------------
def test_model():
    print("="*60)
    print("🚀 EnergoAI v3 Smoke Test")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_small_config()
    model = EnergoAIModel(config).to(device)
    model.eval()

    # Создаём искусственный вход
    batch_size = 2
    seq_len = 64
    vocab_size = config.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    # Маска: 80% реальных токенов, остальное паддинг
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool).to(device)
    for b in range(batch_size):
        pad_start = torch.randint(seq_len//2, seq_len, (1,)).item()
        mask[b, pad_start:] = False
        input_ids[b, pad_start:] = config.pad_token_id

    print(f"Input shape: {input_ids.shape}, device: {device}")
    print(f"Mask sum: {mask.sum(dim=1)}")

    with torch.no_grad():
        logits, diag = model.forward_with_diagnostics(input_ids, mask)

    # ----------- Проверки -----------
    print("\n📊 Диагностика:")
    for k, v in diag.items():
        if isinstance(v, list):
            print(f"  {k}: (list len={len(v)})")
            if v and isinstance(v[0], dict):
                for i, d in enumerate(v):
                    print(f"    loop {i}: Δ_mean={d.get('Δ_mean', '?'):.3f}, "
                          f"h_depth_norm={d.get('h_depth_norm', '?'):.3f}, "
                          f"delta_norm={d.get('delta_norm', '?'):.3f}, "
                          f"decay={d.get('decay', '?'):.3f}")
        else:
            print(f"  {k}: {v}")

    # Проверка на NaN/Inf
    assert not torch.isnan(logits).any(), "❌ Логиты содержат NaN!"
    assert not torch.isinf(logits).any(), "❌ Логиты содержат Inf!"
    print("\n✅ Forward pass без NaN/Inf")

    # Нормы должны быть разумными (не > 100)
    core_norm = diag['core_norm_after_ssm']
    assert core_norm < 100, f"❌ core_norm слишком большая: {core_norm}"
    assert diag['coda_norm'] < 100, f"❌ coda_norm слишком большая: {diag['coda_norm']}"
    print("✅ Нормы в допустимых пределах")

    # Градиенты (дополнительно)
    model.train()
    logits, _ = model.forward_with_diagnostics(input_ids, mask)
    loss = logits.mean()
    loss.backward()
    # Проверим, что градиенты не NaN у основных параметров
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"❌ NaN градиент у {name}!")
            break
    else:
        print("✅ Градиенты без NaN")

    return diag, model

# ------------------- Визуализация -------------------
def plot_diagnostics(diag):
    loop_stats = diag.get('loop_stats', [])
    if not loop_stats:
        print("Нет данных для визуализации")
        return

    loops = list(range(len(loop_stats)))
    h_norms = [s.get('h_depth_norm', 0) for s in loop_stats]
    delta_norms = [s.get('delta_norm', 0) for s in loop_stats]
    delta_means = [s.get('Δ_mean', 0) for s in loop_stats]
    decays = [s.get('decay', 1.0) for s in loop_stats]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(loops, h_norms, 'o-', label='h_depth norm')
    axes[0, 0].set_title('Глубинное состояние по циклам')
    axes[0, 0].set_xlabel('Loop')
    axes[0, 0].set_ylabel('Norm')
    axes[0, 0].grid(True)

    axes[0, 1].plot(loops, delta_norms, 's-', color='orange', label='delta norm')
    axes[0, 1].set_title('Норма Δ-добавки SSM')
    axes[0, 1].set_xlabel('Loop')
    axes[0, 1].set_ylabel('Norm')
    axes[0, 1].grid(True)

    axes[1, 0].plot(loops, delta_means, '^-', color='green', label='Δ mean')
    axes[1, 0].set_title('Среднее Δ (шаг дискретизации)')
    axes[1, 0].set_xlabel('Loop')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].grid(True)

    axes[1, 1].plot(loops, decays, 'D-', color='red', label='decay')
    axes[1, 1].set_title('Коэффициент затухания состояния')
    axes[1, 1].set_xlabel('Loop')
    axes[1, 1].set_ylabel('Decay')
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].grid(True)

    plt.suptitle("Динамика глубинных итераций EnergoAI v3", fontsize=14)
    plt.tight_layout()
    plt.savefig("test_v3_diagnostics.png")
    print("📈 График сохранён в test_v3_diagnostics.png")
    plt.show()

# ------------------- main -------------------
if __name__ == "__main__":
    diag, model = test_model()
    plot_diagnostics(diag)
    print("\n🎉 Тест завершён успешно. Модель стабильна.")