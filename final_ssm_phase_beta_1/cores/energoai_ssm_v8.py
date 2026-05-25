# energoai_v8.py
"""
EnergoAI v8: Balanced Trinity — Wide SSM + Equal Blocks
- Prelude: ~4M
- SSM: ~4M (wide scan N=2048, compression K=64)
- Transformer: ~4M (works on K=64)
- Coda: ~2M
Total: ~14M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==========================================
# 1. RMSNorm
# ==========================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


# ==========================================
# 2. RoPE
# ==========================================
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RoPE(nn.Module):
    def __init__(self, dim, max_seq=8192, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_seq, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos()[None, None, :, :])
        self.register_buffer("sin", emb.sin()[None, None, :, :])

    def get(self, seq_len, device, dtype):
        return (
            self.cos[:, :, :seq_len, :].to(device, dtype),
            self.sin[:, :, :seq_len, :].to(device, dtype)
        )


# ==========================================
# 3. PRELUDE — 4 слоя, ~4M
# ==========================================
class PreludeAttention(nn.Module):
    """Single-head attention (simpler, fewer params)"""

    def __init__(self, D, num_heads):
        super().__init__()
        assert D % num_heads == 0, f"D={D} not divisible by num_heads={num_heads}"
        self.D = D
        self.H = num_heads
        self.dh = D // num_heads

        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.o_proj = nn.Linear(D, D, bias=False)
        self.rope = RoPE(self.dh)
        self.gate = nn.Linear(D, D, bias=True)

        nn.init.xavier_uniform_(self.qkv.weight, gain=0.5)
        nn.init.xavier_uniform_(self.o_proj.weight, gain=0.3)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 2.0)

    def forward(self, x, mask=None):
        B, L, D = x.shape

        qkv = self.qkv(x).view(B, L, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, L, dh]

        cos, sin = self.rope.get(L, x.device, x.dtype)
        q, k = apply_rotary(q, k, cos, sin)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dh)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)

        out = torch.matmul(attn, v)  # [B, H, L, dh]
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.o_proj(out)

        gate = torch.sigmoid(self.gate(x))
        return out * gate


class PreludeMLP(nn.Module):
    def __init__(self, D, expansion=4):
        super().__init__()
        I = D * expansion
        self.gate = nn.Linear(D, I, bias=False)
        self.up = nn.Linear(D, I, bias=False)
        self.down = nn.Linear(I, D, bias=False)

        nn.init.xavier_uniform_(self.gate.weight, gain=0.5)
        nn.init.xavier_uniform_(self.up.weight, gain=0.5)
        nn.init.xavier_uniform_(self.down.weight, gain=0.2)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class PreludeLayer(nn.Module):
    def __init__(self, D, num_heads, expansion=4):
        super().__init__()
        self.norm1 = RMSNorm(D)
        self.attn = PreludeAttention(D, num_heads)
        self.norm2 = RMSNorm(D)
        self.mlp = PreludeMLP(D, expansion)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class PreludeBlock(nn.Module):
    """4 слоя, ~4M при D=768, H=8, expansion=4"""

    def __init__(self, D=768, num_layers=4, num_heads=8, expansion=4):
        super().__init__()
        self.layers = nn.ModuleList([
            PreludeLayer(D, num_heads, expansion)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


# ==========================================
# 4. WIDE SSM — ~4M
# ==========================================
class WideSSM(nn.Module):
    """
    Wide selective scan: N=2048, compression K=64
    Single scan, no loops, input-dependent B and C
    """

    def __init__(self, D=768, N=2048, K=64):
        super().__init__()
        self.D = D
        self.N = N
        self.K = K

        # Input: D -> N (wide!)
        self.W_in = nn.Linear(D, N, bias=False)
        nn.init.orthogonal_(self.W_in.weight, gain=0.5)

        # Selective: input-dependent B and C
        self.B_proj = nn.Linear(D, N, bias=False)
        self.C_proj = nn.Linear(D, N, bias=False)

        # Time constants: per-channel, log-spaced
        self.delta = nn.Parameter(torch.linspace(-4.6, 4.6, N))

        # Compression: N -> K (hard bottleneck!)
        self.W_comp = nn.Linear(N, K, bias=False)
        nn.init.xavier_uniform_(self.W_comp.weight, gain=0.3)

        self.diagnostics = {}

    def forward(self, x):
        B, L, D = x.shape

        # State projection
        state = torch.tanh(self.W_in(x))  # [B, L, N]

        # Selective parameters
        B_val = self.B_proj(x)  # [B, L, N]
        C_val = self.C_proj(x)  # [B, L, N]

        # Time constants
        delta = F.softplus(self.delta)  # [N]
        A_bar = torch.exp(-delta)  # [N]

        # Scan: single pass, wide state
        h = torch.zeros(B, self.N, device=x.device, dtype=x.dtype)
        h_list = []

        for t in range(L):
            # Selective update: A_bar * h + B * state
            h = A_bar * h + B_val[:, t] * state[:, t]
            h_list.append(h)

        h_seq = torch.stack(h_list, dim=1)  # [B, L, N]

        # Compression: only path forward!
        compressed = self.W_comp(h_seq)  # [B, L, K]

        # Diagnostics
        with torch.no_grad():
            self.diagnostics = {
                'h_norm': h_seq.norm(dim=-1).mean().item(),
                'h_max': h_seq.abs().max().item(),
                'delta_mean': delta.mean().item(),
                'delta_range': f"[{delta.min():.3f}-{delta.max():.3f}]",
                'A_bar': A_bar.mean().item(),
                'W_comp_rank': min(self.K, self.N),  # max possible
            }

        return compressed, h_seq[:, -1, :]  # [B, L, K], [B, N]


# ==========================================
# 5. TRANSFORMER — ~4M (works on K=64)
# ==========================================
class TransformerLayer(nn.Module):
    """Single layer, works on compressed K dimensions"""

    def __init__(self, K=64, num_heads=8, expansion=8):
        super().__init__()
        assert K % num_heads == 0, f"K={K} not divisible by num_heads={num_heads}"
        self.K = K
        self.H = num_heads
        self.dh = K // num_heads

        self.qkv = nn.Linear(K, 3 * K, bias=False)
        self.o_proj = nn.Linear(K, K, bias=False)
        self.rope = RoPE(self.dh)

        # FFN: K -> 8K -> K (rich because K is small!)
        I = K * expansion
        self.ffn_gate = nn.Linear(K, I, bias=False)
        self.ffn_up = nn.Linear(K, I, bias=False)
        self.ffn_down = nn.Linear(I, K, bias=False)

        self.norm1 = RMSNorm(K)
        self.norm2 = RMSNorm(K)

        for p in [self.qkv, self.o_proj, self.ffn_gate, self.ffn_up]:
            nn.init.xavier_uniform_(p.weight, gain=0.5)
        nn.init.xavier_uniform_(self.ffn_down.weight, gain=0.2)

    def forward(self, x, mask=None):
        B, L, K = x.shape

        # Attention
        residual = x
        x = self.norm1(x)

        qkv = self.qkv(x).view(B, L, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        cos, sin = self.rope.get(L, x.device, x.dtype)
        q, k = apply_rotary(q, k, cos, sin)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dh)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, K)
        out = self.o_proj(out)

        x = residual + out

        # FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn_down(F.silu(self.ffn_gate(x)) * self.ffn_up(x))
        x = residual + x

        return x


class TransformerBlock(nn.Module):
    def __init__(self, K=64, num_layers=1, num_heads=8, expansion=8):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(K, num_heads, expansion)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


# ==========================================
# 6. CODA — ~2M
# ==========================================
class CodaLayer(nn.Module):
    def __init__(self, D, state_dim, expansion=4):
        super().__init__()
        I = D * expansion

        self.norm = RMSNorm(D)
        self.gate = nn.Linear(D, I, bias=False)
        self.up = nn.Linear(D, I, bias=False)
        self.down = nn.Linear(I, D, bias=False)
        self.res_gate = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

        # State bias
        self.state_proj = nn.Linear(state_dim, D, bias=False)

        nn.init.xavier_uniform_(self.gate.weight, gain=0.1)
        nn.init.xavier_uniform_(self.up.weight, gain=0.1)
        nn.init.xavier_uniform_(self.down.weight, gain=0.01)
        nn.init.xavier_uniform_(self.state_proj.weight, gain=1.5)

    def forward(self, x, state_bias):
        residual = x
        x = self.norm(x)
        x = x + state_bias

        mlp_out = self.down(F.silu(self.gate(x)) * self.up(x))

        # Soft clamp
        mlp_norm = mlp_out.norm(dim=-1, keepdim=True)
        scale = (50.0 / mlp_norm).clamp(max=1.0)
        mlp_out = mlp_out * scale

        return residual + torch.sigmoid(self.res_gate) * mlp_out


class CodaBlock(nn.Module):
    def __init__(self, K=64, D=768, state_dim=2048, num_layers=2, expansion=4):
        super().__init__()

        # K -> D expansion
        self.expand = nn.Linear(K, D, bias=False)
        nn.init.xavier_uniform_(self.expand.weight, gain=0.5)

        # Layers
        self.layers = nn.ModuleList([
            CodaLayer(D, state_dim, expansion)
            for _ in range(num_layers)
        ])

    def forward(self, x, h_final):
        # x: [B, L, K] -> [B, L, D]
        x = self.expand(x)

        # State bias from SSM final state
        state_bias = h_final.unsqueeze(1)  # [B, 1, D] — will broadcast

        for layer in self.layers:
            x = layer(x, state_bias)

        return x


# ==========================================
# 7. FULL MODEL
# ==========================================
class EnergoAIModelV8(nn.Module):
    def __init__(self, config=None):
        super().__init__()

        # Defaults
        cfg = {
            'D': 768,
            'N': 2048,
            'K': 64,
            'V': 32000,
            'prelude_layers': 4,
            'prelude_heads': 8,
            'prelude_expansion': 4,
            'transformer_layers': 1,
            'transformer_heads': 8,
            'transformer_expansion': 8,
            'coda_layers': 2,
            'coda_expansion': 4,
        }
        if config:
            cfg.update(config)

        self.D = cfg['D']
        self.N = cfg['N']
        self.K = cfg['K']

        # Embedding
        self.embed = nn.Embedding(cfg['V'], self.D)

        # Trinity
        self.prelude = PreludeBlock(
            D=self.D,
            num_layers=cfg['prelude_layers'],
            num_heads=cfg['prelude_heads'],
            expansion=cfg['prelude_expansion']
        )

        self.ssm = WideSSM(D=self.D, N=self.N, K=self.K)

        self.transformer = TransformerBlock(
            K=self.K,
            num_layers=cfg['transformer_layers'],
            num_heads=cfg['transformer_heads'],
            expansion=cfg['transformer_expansion']
        )

        self.coda = CodaBlock(
            K=self.K,
            D=self.D,
            state_dim=self.N,
            num_layers=cfg['coda_layers'],
            expansion=cfg['coda_expansion']
        )

        # Output
        self.norm_final = RMSNorm(self.D)
        self.head = nn.Linear(self.D, cfg['V'], bias=False)
        self.head.weight = self.embed.weight  # Tie

        # Phase for curriculum
        self.phase = 0

        # Report
        self._report_params()

    def _report_params(self):
        blocks = {
            'Prelude': self.prelude,
            'SSM': self.ssm,
            'Transformer': self.transformer,
            'Coda': self.coda,
        }
        total = 0
        print(f"\n🧠 EnergoAI v8: Balanced Trinity")
        print("=" * 40)
        for name, block in blocks.items():
            p = sum(x.numel() for x in block.parameters())
            total += p
            print(f"  {name:12s}: {p / 1e6:.2f}M")
        print(f"  {'Total':12s}: {total / 1e6:.2f}M")
        print("=" * 40)

    def forward(self, input_ids, mask=None, return_diag=False):
        B, L = input_ids.shape

        # Embed
        x = self.embed(input_ids)  # [B, L, D]

        # Prelude
        x = self.prelude(x, mask)  # [B, L, D]

        # SSM: only path!
        compressed, h_final = self.ssm(x)  # [B, L, K], [B, N]

        # Transformer on compressed
        trans = self.transformer(compressed, mask)  # [B, L, K]

        # Coda
        x = self.coda(trans, h_final)  # [B, L, D]

        # Output
        x = self.norm_final(x)
        logits = self.head(x)  # [B, L, V]

        if return_diag:
            return logits, {'ssm': self.ssm.diagnostics, 'phase': self.phase}
        return logits

    def set_phase(self, phase):
        """0=SSM only, 1=Transformer+Coda, 2=all"""
        self.phase = phase
        # Freeze/unfreeze logic here if needed
        print(f"Phase set to {phase}")


# ==========================================
# 8. SMOKE TEST
# ==========================================
def smoke_test():
    print("\n" + "=" * 50)
    print("🧪 Smoke Test v8")
    print("=" * 50)

    # Small config for test
    model = EnergoAIModelV8({
        'D': 256,
        'N': 512,
        'K': 32,
        'V': 1000,
        'prelude_layers': 2,
        'prelude_heads': 8,  # 256//8=32, ok
        'prelude_expansion': 4,
        'transformer_layers': 1,
        'transformer_heads': 8,  # 32//8=4, ok
        'transformer_expansion': 8,
        'coda_layers': 2,
        'coda_expansion': 4,
    })

    x = torch.randint(0, 1000, (2, 64))

    # Forward
    logits, diag = model(x, return_diag=True)

    print(f"\n✅ Forward pass: {logits.shape}")
    print(f"   SSM h_norm: {diag['ssm']['h_norm']:.2f}")
    print(f"   SSM delta: {diag['ssm']['delta_mean']:.3f}")
    print(f"   SSM A_bar: {diag['ssm']['A_bar']:.4f}")

    # Check gradients
    loss = logits.mean()
    loss.backward()

    has_grad = {
        'Prelude': any(p.grad is not None for p in model.prelude.parameters()),
        'SSM': any(p.grad is not None for p in model.ssm.parameters()),
        'Transformer': any(p.grad is not None for p in model.transformer.parameters()),
        'Coda': any(p.grad is not None for p in model.coda.parameters()),
    }
    print(f"\n   Gradients: {has_grad}")

    print("\n✅ Smoke test PASSED")
    return model


if __name__ == "__main__":
    smoke_test()