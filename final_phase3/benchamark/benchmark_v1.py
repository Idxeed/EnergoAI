# benchmark_v3.py
"""
Обучение EnergoAI v3 + бенчмарк глубинных итераций.
Измеряет перплексию на тестовом тексте при разном числе циклов (0,1,3,5).
Без изменений в energoai_ssm_v5.py.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import requests

from final_phase3.core.energoai_ssm_v3 import EnergoAIModel, SafeConfig

# ------------------------------
# 1. Датасет (как в обучении)
# ------------------------------
class TextDataset(Dataset):
    def __init__(self, text, seq_length=64):
        self.seq_length = seq_length
        chars = sorted(list(set(text)))
        self.stoi = {ch: i+3 for i,ch in enumerate(chars)}
        self.stoi['<pad>']=0; self.stoi['<bos>']=1; self.stoi['<eos>']=2
        self.itos = {i:s for s,i in self.stoi.items()}
        self.data = [self.stoi.get(ch,2) for ch in text]

    def __len__(self):
        return max(0, len(self.data)-self.seq_length)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.seq_length], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.seq_length+1], dtype=torch.long)
        return x,y

def download_tiny_shakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = requests.get(url).text
    print(f"Загружено {len(text):,} символов")
    return text

# ------------------------------
# 2. Обучение модели (150 шагов)
# ------------------------------
def train_model(device):
    config = SafeConfig({
        "hidden_size":256, "vocab_size":257, "num_attention_heads":4,
        "num_key_value_heads":4, "intermediate_size":512, "rms_norm_eps":1e-6,
        "rope_theta":10000.0, "attention_bias":False,
        "pad_token_id":0, "bos_token_id":1, "eos_token_id":2,
    })
    text = download_tiny_shakespeare()
    split = int(len(text)*0.9)
    train_text = text[:split]
    test_text = text[split:]
    print(f"Train: {len(train_text)} символов, Test: {len(test_text)} символов")

    train_dataset = TextDataset(train_text, seq_length=64)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)

    model = EnergoAIModel(config).to(device)
    print(f"Параметров: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9,0.95), weight_decay=0.01, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lr_lambda=lambda step: min(1.0, step/50) if step<50 else 1.0)

    model.train()
    step = 0
    total_steps = 150
    for epoch in range(3):
        for input_ids, targets in train_loader:
            if step >= total_steps: break
            input_ids = input_ids.to(device); targets = targets.to(device)
            mask = (input_ids != 0).long()
            logits, _ = model.forward_with_diagnostics(input_ids, mask)
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1), ignore_index=0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            if step%50 == 0:
                print(f"Шаг {step}, loss: {loss.item():.3f}")
        if step >= total_steps: break
    return model, test_text, config

# ------------------------------
# 3. Бенчмарк перплексии
# ------------------------------
def evaluate_perplexity(model, text, config, num_loops, seq_length=64):
    dataset = TextDataset(text, seq_length)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=True)
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for input_ids, targets in loader:
            input_ids = input_ids.to(next(model.parameters()).device)
            targets = targets.to(input_ids.device)
            mask = (input_ids != 0).long()
            # ВАЖНО: очищаем чекпоинты, чтобы избежать несовпадения batch-размеров
            model.core_block.checkpoints = []
            logits, _ = model.forward_with_diagnostics(input_ids, mask)
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1), ignore_index=0, reduction='sum')
            total_loss += loss.item()
            total_tokens += (targets != 0).sum().item()
    avg_loss = total_loss / total_tokens
    return torch.exp(torch.tensor(avg_loss)).item()

# ------------------------------
# 4. Основной блок
# ------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    model, test_text, config = train_model(device)

    loops_to_test = [0, 1, 3, 5]
    results = {}
    for loops in loops_to_test:
        model.num_loops = loops
        ppl = evaluate_perplexity(model, test_text, config, loops)
        results[loops] = ppl
        print(f"num_loops={loops}: Perplexity = {ppl:.2f}")

    print("\nРезультаты:")
    print("Циклы | Перплексия")
    for k,v in results.items():
        print(f"  {k}   | {v:.2f}")

    plt.figure(figsize=(6,4))
    plt.plot(loops_to_test, [results[l] for l in loops_to_test], 'o-', linewidth=2, markersize=8)
    plt.xlabel('Число циклов SSM')
    plt.ylabel('Перплексия (ниже = лучше)')
    plt.title('Качество vs глубина (EnergoAI v3)')
    plt.grid(True)
    plt.savefig('benchmark_depth.png')
    print("График сохранён: benchmark_depth.png")
    plt.show()

    # Генерация примера
    print("\nПример сгенерированного текста (стартовая фраза 'FIRST'):")
    start_text = "FIRST"
    start_ids = torch.tensor([[TextDataset(test_text).stoi.get(ch,2) for ch in start_text]], device=device)
    for loops in [0,5]:
        model.num_loops = loops
        model.eval()
        with torch.no_grad():
            input_ids = start_ids.clone()
            for _ in range(40):
                mask = (input_ids != 0).long()
                logits, _ = model(input_ids, mask)
                next_token = torch.argmax(logits[:,-1,:], dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)
            generated = ''.join([TextDataset(test_text).itos.get(t.item(),'?') for t in input_ids[0]])
            print(f"  loops={loops}: {generated}")
