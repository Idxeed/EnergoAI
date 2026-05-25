# EnergoAI

**A Hybrid State Space + Transformer Architecture for Efficient Language Modeling**

EnergoAI is a research-oriented series of hybrid neural network architectures that combine **State Space Models (SSM)** with modern Transformer components. The project explores efficient memory compression, curriculum learning, and scalable long-context modeling.

Developed from scratch in PyTorch with a strong focus on architectural innovation and training stability.

---

## 🎯 Project Goals

- Create a highly efficient hybrid LLM architecture that outperforms traditional Transformers in memory usage and long-context handling.
- Explore the synergy between **Selective State Space Models** and **Transformer layers**.
- Build production-ready, debuggable, and curriculum-trainable models.
- Push the boundaries of efficient AI systems suitable for real-world deployment.

---

## ✨ Key Features

- **Wide SSM Compressor** — powerful memory bottleneck with input-dependent Δ, forget gates, and selective scan (parallel-ready)
- **Curriculum Learning** — progressive training from SSM-only to full hybrid mode
- **Latent GQA Attention** — efficient Grouped Query Attention with KV compression
- **State-Conditioned Coda Block** — final refinement using compressed memory state
- **Flexible Output Head** — memory-aware logit prediction
- **Extensive Diagnostics** — monitoring of rank, delta diversity, gradient norms, etc.
- **Robust Training Pipeline** — proper masking, dropout, weight initialization, and phase control

---

## 📊 Architecture (v9.2)
Input → Embedding
↓
[Prelude] — 2-4 layers of gated attention + MLP
↓
[Wide SSM] — D → N (wide state) → K (compressed)
↓
[Transformer] — Latent GQA layers (activated via curriculum)
↓
[Coda Block] — State-conditioned refinement
↓
[Flexible Head] — Base + Memory Adapter
text**Core Innovation**: SSM acts as an **obligatory compressor**, significantly reducing sequence length before the Transformer stage.

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/energoai.git
cd energoai
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
Smoke Test
Pythonfrom energoai import EnergoAIModelV9, EnergoAIConfig

config = EnergoAIConfig(
    D=256,
    N=512,
    K=32,
    V=1000,
    prelude_layers=2,
    transformer_layers=1
)

model = EnergoAIModelV9(config)
model.set_phase(1)  # Enable full hybrid mode

x = torch.randint(0, 1000, (2, 64))
logits, diagnostics = model(x, return_diag=True)

print(f"Output shape: {logits.shape}")
print(f"SSM h_norm: {diagnostics['h_norm']:.4f}")
```

📈 Current Status

Version: v9.2 (Fixed & Hardened)
Model Size: Configurable (currently testing 14M–400M+ parameter range)
Training: Curriculum-based (Phase 0 → Phase 1)
Focus: Efficiency, Stability, Long Context

Next Milestones:

Full parallel selective scan (torch.scan / custom kernel)
Larger scale pretraining (124M → 1B+)
Quantization & inference optimization
Open weights release


🛠 Tech Stack

Framework: PyTorch
Core: State Space Models, Transformer, RMSNorm, RoPE, GQA
Tools: Torch.compile, mixed precision, gradient checkpointing (planned)


📄 License
This project is for research and educational purposes. All rights reserved.

📬 Contact & Links

Author: Timur Averin
Email: murzabekovr5@gmail.com

"Building efficient intelligence from first principles."
