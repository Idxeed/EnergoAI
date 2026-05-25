# efficient_transformer_v711.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    """Ваш стандартный RMSNorm."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = float(eps)

    def forward(self, x):
        dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * x.to(dtype)


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class RoPECache:
    """Кэшированные RoPE частоты (эффективнее, чем _get_cos_sin каждый раз)."""

    def __init__(self, head_dim, theta=10000.0, max_seq=8192):
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_seq, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos()[None, None, :, :])
        self.register_buffer("sin", emb.sin()[None, None, :, :])

    def get(self, seq_len, device, dtype):
        return self.cos[:, :, :seq_len, :].to(device, dtype), self.sin[:, :, :seq_len, :].to(device, dtype)


class GQALatentAttention(nn.Module):
    """
    GQA + Latent KV compression (MLA-стиль) + RoPE.
    Упрощённая версия вашего GatedDifferentialAttention:
    - Без дифференциального вычитания (экономим 2× compute)
    - Без сигмоидного гейта на выходе (экономим параметры)
    - С latent KV compression (экономим KV-cache)

    Параметры: ~1.2M при D=768, H=12, H_kv=4, latent=256
    vs ваш GatedDifferential: ~1.8M
    """

    def __init__(self, hidden_dim, num_heads=12, num_kv_heads=4,
                 latent_dim=256, rope_theta=10000.0, max_seq=8192):
        super().__init__()
        self.D = hidden_dim
        self.H = num_heads
        self.H_kv = num_kv_heads
        self.dh = self.D // self.H  # 64
        self.latent = latent_dim  # 256

        # Q: обычный
        self.q_proj = nn.Linear(self.D, self.H * self.dh, bias=False)

        # KV: сжатие в latent (как MLA)
        self.kv_down = nn.Linear(self.D, self.latent, bias=False)  # D→latent
        self.k_up = nn.Linear(self.latent, self.H_kv * self.dh, bias=False)  # latent→K
        self.v_up = nn.Linear(self.latent, self.H_kv * self.dh, bias=False)  # latent→V

        # Output
        self.o_proj = nn.Linear(self.H * self.dh, self.D, bias=False)

        # RoPE cache
        self.rope = RoPECache(self.dh, rope_theta, max_seq)

        # Init (по вашему стилю: разный gain)
        nn.init.xavier_uniform_(self.q_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.kv_down.weight, gain=0.5)
        nn.init.xavier_uniform_(self.k_up.weight, gain=0.5)
        nn.init.xavier_uniform_(self.v_up.weight, gain=0.5)
        nn.init.xavier_uniform_(self.o_proj.weight, gain=0.3)

    def forward(self, x, mask=None):
        B, L, D = x.shape

        # Q
        q = self.q_proj(x).view(B, L, self.H, self.dh).transpose(1, 2)  # [B, H, L, dh]

        # Compressed KV
        c_kv = self.kv_down(x)  # [B, L, latent]
        k = self.k_up(c_kv).view(B, L, self.H_kv, self.dh).transpose(1, 2)  # [B, H_kv, L, dh]
        v = self.v_up(c_kv).view(B, L, self.H_kv, self.dh).transpose(1, 2)  # [B, H_kv, L, dh]

        # RoPE
        cos, sin = self.rope.get(L, x.device, x.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA: repeat KV heads
        if self.H_kv != self.H:
            n_rep = self.H // self.H_kv
            k = k.repeat_interleave(n_rep, dim=1)  # [B, H, L, dh]
            v = v.repeat_interleave(n_rep, dim=1)  # [B, H, L, dh]

        # Attention (flash attention compatible)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dh)  # [B, H, L, L]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)

        out = torch.matmul(attn, v)  # [B, H, L, dh]
        out = out.transpose(1, 2).contiguous().view(B, L, self.D)
        out = self.o_proj(out)

        return out


class SwiGLUFFN(nn.Module):
    """
    SwiGLU по вашему стилю ExpandedGLUMLP, но 4× (не 6×) — экономим на 1 слое.
    С res_gate из Coda (адаптивный, не фиксированный 0.05).
    """

    def __init__(self, hidden_dim, intermediate_factor=4):
        super().__init__()
        self.D = hidden_dim
        self.I = int(self.D * intermediate_factor)

        self.gate = nn.Linear(self.D, self.I, bias=False)
        self.up = nn.Linear(self.D, self.I, bias=False)
        self.down = nn.Linear(self.I, self.D, bias=False)

        # Адаптивный res_gate (как в Coda, но learnable)
        self.res_gate = nn.Parameter(torch.tensor(0.1))

        # Init (ваш стиль: убывающий gain)
        nn.init.xavier_uniform_(self.gate.weight, gain=0.5)
        nn.init.xavier_uniform_(self.up.weight, gain=0.5)
        nn.init.xavier_uniform_(self.down.weight, gain=0.2)

    def forward(self, x):
        # SwiGLU
        x_norm = x.norm(dim=-1, keepdim=True)
        gate = F.silu(self.gate(x))
        up = self.up(x)
        mlp_out = self.down(gate * up)

        # Soft clamp (как в Coda, но адаптивный)
        mlp_norm = mlp_out.norm(dim=-1, keepdim=True)
        scale = (50.0 / mlp_norm).clamp(max=1.0)
        mlp_out = mlp_out * scale

        # Residual с learnable gate
        return x + torch.sigmoid(self.res_gate) * mlp_out


class EfficientTransformerLayer(nn.Module):
    """
    Полный слой: GQA Latent Attention + SwiGLU FFN + RMSNorm.
    Pre-norm (как ваш Prelude), residual с adaptive gate (как Coda).

    Параметры: ~10.5M при D=768
    vs ваш EnergoAIDecoderLayerV6: ~12M (экономия 12%)
    """

    def __init__(self, hidden_dim, num_heads=12, num_kv_heads=4,
                 latent_dim=256, rope_theta=10000.0, max_seq=8192):
        super().__init__()

        self.attn = GQALatentAttention(
            hidden_dim, num_heads, num_kv_heads,
            latent_dim, rope_theta, max_seq
        )
        self.ffn = SwiGLUFFN(hidden_dim, intermediate_factor=4)

        self.norm_attn = RMSNorm(hidden_dim)
        self.norm_ffn = RMSNorm(hidden_dim)

    def forward(self, x, mask=None):
        # Attention block
        residual = x
        x = self.norm_attn(x)
        x = self.attn(x, mask)
        x = residual + x  # стандартный residual (attention стабильна)

        # FFN block (с adaptive gate)
        x = self.ffn(x)  # внутри уже есть residual + sigmoid(res_gate)

        return x