# energoai_ssm_v711.py
"""
EnergoAI SSM v7.1.1 — SSM как обязательный компрессор + Transformer слой.
- SSM Compressor: D→N→D bottleneck, единственный путь, без skip
- Transformer: 1 слой GQA Latent Attention + SwiGLU (эффективный)
- Prelude: 2 слоя (упрощённый)
- Coda: 2 слоя без checkpoint attention (он теперь в transformer)
- Curriculum-ready: флаги для фазового обучения
- Диагностика живости SSM: grad norms, rank W_in, loss_aux
"""

import math
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. CONFIG & PRIMITIVES
# ==========================================
class SafeConfig:
    def __init__(self, raw_dict):
        defaults = {
            "pad_token_id": None, "bos_token_id": 1, "eos_token_id": 151645,
            "attention_bias": False, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
            "num_key_value_heads": raw_dict.get("num_attention_heads", 32),
            "hidden_act": "silu", "use_cache": True
        }
        self.__dict__.update(defaults)
        self.__dict__.update({str(k).strip(): v for k, v in raw_dict.items()})

        self.num_prelude_layers = raw_dict.get("num_prelude_layers", 2)
        self.num_coda_layers = raw_dict.get("num_coda_layers", 2)
        self.num_loops = raw_dict.get("num_loops", 4)
        self.state_dim = raw_dict.get("state_dim", 256)
        self.checkpoint_every = raw_dict.get("checkpoint_every", 1024)
        self.num_slots = raw_dict.get("num_slots", 32)

        # v7.1.1 новые параметры
        self.transformer_heads = raw_dict.get("transformer_heads", 12)
        self.transformer_kv_heads = raw_dict.get("transformer_kv_heads", 4)
        self.transformer_latent = raw_dict.get("transformer_latent", 256)
        self.curriculum_phase = raw_dict.get("curriculum_phase", 0)  # 0=SSM only, 1=full

        for k in ["hidden_size", "num_attention_heads", "num_key_value_heads", "intermediate_size", "vocab_size"]:
            if hasattr(self, k):
                setattr(self, k, int(getattr(self, k)))
        for k in ["rope_theta", "rms_norm_eps"]:
            if hasattr(self, k):
                setattr(self, k, float(getattr(self, k)))


class RMSNorm(nn.Module):
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


# ==========================================
# 2. PRELUDE v7.1.1 (упрощённый, 2 слоя)
# ==========================================
class GatedDifferentialAttention(nn.Module):
    """Упрощённая версия из v7.0 — без дифференциального вычитания (экономим)."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.rope_theta = float(config.rope_theta)
        self.attention_bias = bool(config.attention_bias)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=self.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=self.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=self.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=self.attention_bias)

        self.gate_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 2.0)

        inv_freq = 1.0 / (
                    self.rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / float(self.head_dim)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _get_cos_sin(self, seq_len, device, dtype):
        t = torch.arange(int(seq_len), device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None, :, :].to(dtype), emb.sin()[None, None, :, :].to(dtype)

    def forward(self, hidden_states, attention_mask=None):
        bsz, q_len = hidden_states.shape[:2]

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self._get_cos_sin(q_len, hidden_states.device, hidden_states.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        causal_mask = torch.triu(torch.ones(q_len, q_len, device=hidden_states.device, dtype=torch.bool), diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            pad_mask = (~attention_mask.bool()).unsqueeze(1).unsqueeze(2)
            combined_mask = causal_mask | pad_mask
        else:
            combined_mask = causal_mask

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(combined_mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        out = self.o_proj(out)

        gate = torch.sigmoid(self.gate_proj(hidden_states))
        out = out * gate
        return out


class ExpandedGLUMLP(nn.Module):
    def __init__(self, config, intermediate_multiplier=4):
        super().__init__()
        hs = int(config.hidden_size)
        ims = int(config.hidden_size * intermediate_multiplier)
        bias = bool(config.attention_bias)

        self.gate_proj = nn.Linear(hs, ims, bias=bias)
        self.up_proj = nn.Linear(hs, ims, bias=bias)
        self.down_proj = nn.Linear(ims, hs, bias=bias)

        nn.init.xavier_uniform_(self.gate_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.up_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.down_proj.weight, gain=0.2)
        if bias:
            nn.init.zeros_(self.gate_proj.bias)
            nn.init.zeros_(self.up_proj.bias)
            nn.init.zeros_(self.down_proj.bias)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class EnergoAIDecoderLayerV711(nn.Module):
    def __init__(self, config, intermediate_multiplier=4):
        super().__init__()
        hs, eps = int(config.hidden_size), float(config.rms_norm_eps)
        self.input_layernorm = RMSNorm(hs, eps=eps)
        self.self_attn = GatedDifferentialAttention(config)
        self.post_attention_layernorm = RMSNorm(hs, eps=eps)
        self.mlp = ExpandedGLUMLP(config, intermediate_multiplier=intermediate_multiplier)

    def forward(self, x, mask=None):
        x = x + self.self_attn(self.input_layernorm(x), mask)
        return x + self.mlp(self.post_attention_layernorm(x))


# ==========================================
# 3. SSM COMPRESSOR v7.1.1 (единственный путь)
# ==========================================
class SSMCompressor(nn.Module):
    """
    SSM как обязательный компрессор памяти — ПАТЧ v7.1.1a.
    Фиксы:
    1. Логарифмическая инициализация Δ (разнообразие временных масштабов)
    2. Input-dependent Δ (адаптация под вход)
    3. Forget gate (контролируемое забывание)
    4. Curriculum-aware auxiliary loss weight
    """

    def __init__(self, hidden_dim, state_dim, num_loops=4):
        super().__init__()
        self.D = hidden_dim
        self.N = state_dim
        self.num_loops = num_loops

        # === КОМПРЕССИЯ ===
        self.W_in = nn.Linear(self.D, self.N, bias=False)
        nn.init.orthogonal_(self.W_in.weight, gain=0.5)

        self.W_out = nn.Linear(self.N, self.D, bias=False)
        nn.init.xavier_uniform_(self.W_out.weight, gain=0.3)

        # === Δ: база + input-dependent ===
        # Базовый Δ: логарифмическое распределение (0.01 to 100)
        self.delta_base = nn.Parameter(torch.zeros(self.N))
        with torch.no_grad():
            log_delta = torch.linspace(-4.6, 4.6, self.N)  # log(0.01) to log(100)
            self.delta_base.copy_(log_delta)

        # Input-dependent Δ (как в v7.0)
        self.delta_proj = nn.Linear(self.D, self.N, bias=False)
        nn.init.xavier_uniform_(self.delta_proj.weight, gain=0.3)

        # Δ scale (learnable multiplier)
        self.delta_scale = nn.Parameter(torch.tensor(0.0))

        # === FORGET GATE ===
        self.forget_gate = nn.Linear(self.D, self.N, bias=True)
        nn.init.xavier_uniform_(self.forget_gate.weight, gain=0.3)
        nn.init.constant_(self.forget_gate.bias, -1.0)  # Старт: открыт (забываем мало)

        # === STATE PROCESSING ===
        self.state_norm = RMSNorm(self.N, eps=1e-6)
        self.input_gain = nn.Parameter(torch.tensor(0.5))  # Старт выше, чем 0.1

        # === AUXILIARY ===
        self.reconstruction_head = nn.Linear(self.N, self.D, bias=False)
        nn.init.xavier_uniform_(self.reconstruction_head.weight, gain=0.01)

        self.diagnostics = {}

    def forward(self, x, return_aux=False, curriculum_phase=0):
        B, L, D = x.shape

        # === КОМПРЕССИЯ ===
        state = self.input_gain * torch.tanh(self.W_in(x))  # [B, L, N]
        state = self.state_norm(state)

        # === Δ: база + input-dependent ===
        delta_input = self.delta_proj(x)  # [B, L, N]
        delta_logits = self.delta_base.view(1, 1, -1) + delta_input  # [B, L, N]
        delta_logits = torch.clamp(delta_logits, min=-10, max=10)  # защита от переполнения
        Δ = F.softplus(delta_logits) * torch.sigmoid(self.delta_scale)  # [B, L, N]

        A_bar = torch.exp(-Δ)  # [B, L, N]
        A_bar = torch.clamp(A_bar, min=1e-6, max=1.0)

        # === FORGET GATE ===
        G = torch.sigmoid(self.forget_gate(x))  # [B, L, N]
        G = torch.clamp(G, min=1e-6, max=1.0)

        # === SSM SCAN ===
        h = torch.zeros(B, self.N, device=x.device, dtype=x.dtype)
        h_list = []
        # СПИСКИ для диагностики
        delta_list = []
        a_bar_list = []
        g_list = []

        for t in range(L):
            h = G[:, t] * (A_bar[:, t] * h) + (1.0 - G[:, t]) * Δ[:, t] * state[:, t]
            h_list.append(h)
            # Заполняем списки
            delta_list.append(Δ[:, t])
            a_bar_list.append(A_bar[:, t])
            g_list.append(G[:, t])

        h_seq = torch.stack(h_list, dim=1)  # [B, L, N]

        # === ДЕКОМПРЕССИЯ ===
        output = self.W_out(h_seq)  # [B, L, D]

        # === ДИАГНОСТИКА ===
        with torch.no_grad():
            deltas = torch.stack(delta_list, dim=1)  # [B, L, N]
            a_bars = torch.stack(a_bar_list, dim=1)  # [B, L, N]
            gates = torch.stack(g_list, dim=1)  # [B, L, N]

            # Rank W_in
            _, s, _ = torch.svd(self.W_in.weight)
            effective_rank = (s > s[0] * 0.01).sum().item()

            # Δ diversity
            delta_per_channel = deltas.mean(dim=(0, 1))  # [N]
            active_channels = (delta_per_channel > 0.01).sum().item()

            # Min/max с защитой
            delta_min = deltas.min().item()
            delta_max = deltas.max().item()
            a_bar_min = a_bars.min().item()
            a_bar_max = a_bars.max().item()
            g_min = gates.min().item()
            g_max = gates.max().item()

            self.diagnostics = {
                "h_norm": h_seq.norm(dim=-1).mean().item(),
                "h_max": h_seq.abs().max().item(),
                "delta_mean": deltas.mean().item(),
                "delta_min": delta_min,
                "delta_max": delta_max,
                "delta_std": deltas.std().item(),
                "A_bar_mean": a_bars.mean().item(),
                "A_bar_min": a_bar_min,
                "A_bar_max": a_bar_max,
                "G_mean": gates.mean().item(),
                "G_min": g_min,
                "G_max": g_max,
                "W_in_rank": effective_rank,
                "active_channels": active_channels,
                "delta_scale": torch.sigmoid(self.delta_scale).item(),
                "input_gain": self.input_gain.item(),
            }

        if return_aux:
            recon = self.reconstruction_head(h_seq)  # [B, L, D]
            aux_weight = 2.0 if curriculum_phase == 0 else 0.5
            return output, recon, h_seq, aux_weight

        return output, h_seq


# ==========================================
# 4. TRANSFORMER LAYER v7.1.1 (эффективный)
# ==========================================
class RoPECache:
    """Кэшированные RoPE частоты."""

    def __init__(self, head_dim, theta=10000.0, max_seq=8192):
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_seq, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos()[None, None, :, :])
        self.register_buffer("sin", emb.sin()[None, None, :, :])

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)

    def get(self, seq_len, device, dtype):
        return self.cos[:, :, :seq_len, :].to(device, dtype), self.sin[:, :, :seq_len, :].to(device, dtype)


class GQALatentAttention(nn.Module):
    """
    GQA + Latent KV compression (MLA-стиль).
    Параметры: ~1.2M при D=768, H=12, H_kv=4, latent=256
    """

    def __init__(self, hidden_dim, num_heads=12, num_kv_heads=4,
                 latent_dim=256, rope_theta=10000.0, max_seq=8192):
        super().__init__()
        self.D = hidden_dim
        self.H = num_heads
        self.H_kv = num_kv_heads
        self.dh = self.D // self.H
        self.latent = latent_dim

        # Q: обычный
        self.q_proj = nn.Linear(self.D, self.H * self.dh, bias=False)

        # KV: сжатие в latent
        self.kv_down = nn.Linear(self.D, self.latent, bias=False)
        self.k_up = nn.Linear(self.latent, self.H_kv * self.dh, bias=False)
        self.v_up = nn.Linear(self.latent, self.H_kv * self.dh, bias=False)

        # Output
        self.o_proj = nn.Linear(self.H * self.dh, self.D, bias=False)

        # RoPE
        self.rope = RoPECache(self.dh, rope_theta, max_seq)

        # Init
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

        # GQA repeat
        if self.H_kv != self.H:
            n_rep = self.H // self.H_kv
            k = k.repeat_interleave(n_rep, dim=1)  # [B, H, L, dh]
            v = v.repeat_interleave(n_rep, dim=1)  # [B, H, L, dh]

        # Attention scores: [B, H, L, L]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dh)  # [B, H, L, L]

        # === FIX: правильная обработка mask ===
        if mask is not None:
            # mask: [B, L] или [B, 1, L, L] или [1, 1, L, L]
            if mask.dim() == 2:
                # [B, L] → каузальная + padding
                # Сначала каузальная маска
                causal = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
                causal = causal.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]

                # Padding mask: [B, L] → [B, 1, 1, L]
                pad_mask = (~mask.bool()).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]

                # Комбинируем: каузальная ИЛИ padding
                # causal [1,1,L,L] broadcast + pad_mask [B,1,1,L] broadcast
                combined = causal | pad_mask  # [B, 1, L, L]
                scores = scores.masked_fill(combined, float('-inf'))
            elif mask.dim() == 3:
                # [B, 1, L] или [B, H, L]
                mask = mask.unsqueeze(2) if mask.dim() == 3 else mask  # [B, H, 1, L]
                scores = scores.masked_fill(mask == 0, float('-inf'))
            elif mask.dim() == 4:
                # [B, H, L, L] — готовая
                scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)

        out = torch.matmul(attn, v)  # [B, H, L, dh]
        out = out.transpose(1, 2).contiguous().view(B, L, self.D)  # [B, L, D]
        out = self.o_proj(out)

        return out


class SwiGLUFFN(nn.Module):
    """SwiGLU с adaptive residual gate (из Coda)."""

    def __init__(self, hidden_dim, intermediate_factor=4):
        super().__init__()
        self.D = hidden_dim
        self.I = int(self.D * intermediate_factor)

        self.gate = nn.Linear(self.D, self.I, bias=False)
        self.up = nn.Linear(self.D, self.I, bias=False)
        self.down = nn.Linear(self.I, self.D, bias=False)

        self.res_gate = nn.Parameter(torch.tensor(0.1))

        nn.init.xavier_uniform_(self.gate.weight, gain=0.5)
        nn.init.xavier_uniform_(self.up.weight, gain=0.5)
        nn.init.xavier_uniform_(self.down.weight, gain=0.2)

    def forward(self, x):
        gate = F.silu(self.gate(x))
        up = self.up(x)
        mlp_out = self.down(gate * up)

        # Soft clamp
        mlp_norm = mlp_out.norm(dim=-1, keepdim=True)
        scale = (50.0 / mlp_norm).clamp(max=1.0)
        mlp_out = mlp_out * scale

        return x + torch.sigmoid(self.res_gate) * mlp_out


class EfficientTransformerLayer(nn.Module):
    """Полный слой: GQA + SwiGLU + RMSNorm."""

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
        # Attention
        residual = x
        x = self.norm_attn(x)
        x = self.attn(x, mask)
        x = residual + x

        # FFN (с internal residual)
        x = self.ffn(self.norm_ffn(x))

        return x


# ==========================================
# 5. CODA v7.1.1 (упрощённый, без checkpoint attention)
# ==========================================
class CodaBlock(nn.Module):
    def __init__(self, hidden_dim, intermediate):
        super().__init__()
        self.norm = RMSNorm(hidden_dim)
        self.gate = nn.Linear(hidden_dim, intermediate, bias=False)
        self.up = nn.Linear(hidden_dim, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden_dim, bias=False)
        self.res_gate = nn.Parameter(torch.tensor(0.05))

        nn.init.xavier_uniform_(self.gate.weight, gain=0.1)
        nn.init.xavier_uniform_(self.up.weight, gain=0.1)
        nn.init.xavier_uniform_(self.down.weight, gain=0.01)

    def forward(self, hidden_states, state_bias):
        x = self.norm(hidden_states)
        x = x + state_bias
        mlp_out = self.down(F.silu(self.gate(x)) * self.up(x))

        mlp_norm = mlp_out.norm(dim=-1, keepdim=True)
        scale = (10.0 / mlp_norm).clamp(max=1.0)
        mlp_out = mlp_out * scale

        return hidden_states + self.res_gate * mlp_out


class StateConditionedMLP(nn.Module):
    def __init__(self, hidden_dim, state_dim, intermediate_factor=2.5, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.num_layers = num_layers

        intermediate = int(hidden_dim * intermediate_factor)
        self.state_proj = nn.Linear(state_dim, hidden_dim, bias=False)
        nn.init.xavier_uniform_(self.state_proj.weight, gain=1.5)

        # Все слои — базовые (без checkpoint attention, оно теперь в transformer)
        self.layers = nn.ModuleList([
            CodaBlock(hidden_dim, intermediate)
            for _ in range(num_layers)
        ])

        self.diagnostics = {}

    def forward(self, hidden_states, h_depth):
        B, L, D = hidden_states.shape
        state_bias = self.state_proj(h_depth).unsqueeze(1)

        layer_norms = []
        for layer in self.layers:
            hidden_states = layer(hidden_states, state_bias)
            layer_norms.append(round(hidden_states.norm(dim=-1).mean().item(), 3))

        self.diagnostics = {
            'coda_layer_norms': layer_norms,
            'state_bias_norm': round(state_bias.norm(dim=-1).mean().item(), 3),
            'res_gates': [round(float(layer.res_gate.item()), 3) for layer in self.layers],
        }

        return hidden_states


# ==========================================
# 6. OUTPUT HEAD (без изменений)
# ==========================================
class FlexibleOutputHead(nn.Module):
    def __init__(self, config, memory_dim=128):
        super().__init__()
        self.base_head = nn.Linear(int(config.hidden_size), int(config.vocab_size), bias=False)
        self.adapter = nn.Sequential(
            nn.Linear(int(config.hidden_size) + memory_dim, 256),
            nn.SiLU(),
            nn.Linear(256, int(config.vocab_size))
        )
        self.gate_proj = nn.Linear(memory_dim, 1, bias=True)
        nn.init.constant_(self.gate_proj.bias, -1.0)
        with torch.no_grad():
            for layer in self.adapter:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.01)
                    if layer.bias is not None:
                        layer.bias.zero_()

    def forward(self, hidden_states, memory_state):
        bsz, seq_len = hidden_states.shape[:2]
        base = self.base_head(hidden_states)
        mem_exp = memory_state.unsqueeze(1).expand(-1, seq_len, -1)
        delta = self.adapter(torch.cat([hidden_states, mem_exp], dim=-1))
        gate = torch.sigmoid(self.gate_proj(memory_state)).view(bsz, 1, 1)
        return base + gate * delta


# ==========================================
# 7. ENERGOAI MODEL v7.1.1
# ==========================================
class EnergoAIModelV711(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        pad_idx = int(config.pad_token_id) if config.pad_token_id is not None else None
        self.embed_tokens = nn.Embedding(int(config.vocab_size), int(config.hidden_size), padding_idx=pad_idx)

        # === PRELUDE: 2 слоя (упрощённый) ===
        self.prelude = nn.ModuleList([
            EnergoAIDecoderLayerV711(config, intermediate_multiplier=4)
            for _ in range(int(config.num_prelude_layers))
        ])

        # === SSM COMPRESSOR: единственный путь ===
        self.ssm_compressor = SSMCompressor(
            hidden_dim=int(config.hidden_size),
            state_dim=int(config.state_dim),
            num_loops=int(config.num_loops)
        )

        # === TRANSFORMER: 1 слой (после SSM) ===
        self.transformer = EfficientTransformerLayer(
            hidden_dim=int(config.hidden_size),
            num_heads=int(config.transformer_heads),
            num_kv_heads=int(config.transformer_kv_heads),
            latent_dim=int(config.transformer_latent),
            rope_theta=float(config.rope_theta),
            max_seq=8192
        )

        # === CODA: 2 слоя (упрощённый) ===
        self.coda = StateConditionedMLP(
            hidden_dim=int(config.hidden_size),
            state_dim=int(config.state_dim),
            intermediate_factor=2.5,
            num_layers=int(config.num_coda_layers)
        )

        # === Curriculum control ===
        self.curriculum_phase = nn.Parameter(torch.tensor(int(config.curriculum_phase)), requires_grad=False)
        self.transformer_gate = nn.Parameter(torch.tensor(0.0))  # 0=закрыт (фаза 1), 1=открыт (фаза 2)
        nn.init.constant_(self.transformer_gate, 0.0)

        self.norm = RMSNorm(int(config.hidden_size), eps=float(config.rms_norm_eps))
        self.output_head = FlexibleOutputHead(config, memory_dim=int(config.state_dim))
        self.output_head.base_head.weight = self.embed_tokens.weight

        # Auxiliary loss weight
        self.lambda_aux = 0.1

    def forward(self, input_ids, attention_mask=None):
        logits, _ = self.forward_with_diagnostics(input_ids, attention_mask)
        return logits

    def forward_with_diagnostics(self, input_ids, attention_mask=None):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        hidden_states = self.embed_tokens(input_ids)
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            if attention_mask.dim() == 2 and attention_mask.shape[0] != bsz:
                attention_mask = attention_mask.expand(bsz, -1)

        # === PRELUDE ===
        for layer in self.prelude:
            hidden_states = layer(hidden_states, attention_mask)
        prelude_norm = hidden_states.norm(dim=-1).mean().item()

        # === SSM COMPRESSOR ===
        ssm_output, recon, h_seq, aux_weight = self.ssm_compressor(
            hidden_states,
            return_aux=True,
            curriculum_phase=self.curriculum_phase.item()
        )

        # === FIX: вычисляем ssm_norm ===
        ssm_norm = ssm_output.norm(dim=-1).mean().item()

        # Auxiliary loss
        loss_aux = F.mse_loss(recon, hidden_states)
        loss_aux_weighted = aux_weight * loss_aux

        # Диагностика SSM
        ssm_diag = dict(self.ssm_compressor.diagnostics)
        ssm_diag["loss_aux"] = loss_aux.item()
        ssm_diag["loss_aux_weighted"] = loss_aux_weighted.item()
        ssm_diag["aux_weight"] = aux_weight

        # === TRANSFORMER ===
        gate = torch.sigmoid(self.transformer_gate)

        if self.curriculum_phase.item() >= 1:
            transformer_output = self.transformer(ssm_output, attention_mask)
            hidden_states = gate * transformer_output + (1 - gate) * ssm_output
        else:
            hidden_states = ssm_output
            transformer_output = None

        transformer_norm = hidden_states.norm(dim=-1).mean().item() if transformer_output is not None else 0.0

        # === CODA ===
        h_depth = h_seq[:, -1, :]  # [B, N]
        hidden_states = self.coda(hidden_states, h_depth)
        coda_norm = hidden_states.norm(dim=-1).mean().item()

        # === OUTPUT ===
        hidden_states = self.norm(hidden_states)
        logits = self.output_head(hidden_states, h_depth)

        # === ДИАГНОСТИКА ===
        diagnostics = {
            "prelude_norm": round(prelude_norm, 3),
            "ssm_norm": round(ssm_norm, 3),  # ← теперь определено
            "ssm_h_norm": round(ssm_diag["h_norm"], 3),
            "ssm_delta_mean": round(ssm_diag["delta_mean"], 3),
            "ssm_delta_std": round(ssm_diag.get("delta_std", 0.0), 3),
            "ssm_delta_min": round(ssm_diag["delta_min"], 3),
            "ssm_delta_max": round(ssm_diag["delta_max"], 3),
            "ssm_delta_range": f"[{ssm_diag['delta_min']:.3f}-{ssm_diag['delta_max']:.3f}]",
            "ssm_A_bar": round(ssm_diag["A_bar_mean"], 4),
            "ssm_A_bar_range": f"[{ssm_diag.get('A_bar_min', 0):.4f}-{ssm_diag.get('A_bar_max', 0):.4f}]",
            "ssm_G_mean": round(ssm_diag.get("G_mean", 0.0), 4),
            "ssm_G_range": f"[{ssm_diag.get('G_min', 0):.4f}-{ssm_diag.get('G_max', 0):.4f}]",
            "ssm_W_in_rank": ssm_diag["W_in_rank"],
            "ssm_active_channels": ssm_diag.get("active_channels", 0),
            "ssm_delta_scale": round(ssm_diag.get("delta_scale", 0.0), 4),
            "ssm_input_gain": round(ssm_diag.get("input_gain", 0.0), 4),
            "ssm_loss_aux": round(ssm_diag["loss_aux"], 4),
            "ssm_loss_aux_weighted": round(ssm_diag["loss_aux_weighted"], 4),
            "transformer_gate": round(gate.item(), 4),
            "transformer_norm": round(transformer_norm, 3),
            "coda_norm": round(coda_norm, 3),
            "coda_res_gates": self.coda.diagnostics.get('res_gates', []),
            "coda_layer_norms": self.coda.diagnostics.get('coda_layer_norms', []),
            "h_depth_norm": round(h_depth.norm(dim=-1).mean().item(), 3),
        }

        return logits.view(bsz, seq_len, -1), diagnostics, loss_aux_weighted

    def set_curriculum_phase(self, phase):
        """0 = SSM only, 1 = SSM + Transformer"""
        self.curriculum_phase.data = torch.tensor(phase)
        if phase == 0:
            nn.init.constant_(self.transformer_gate, -5.0)  # sigmoid(-5) ≈ 0
        else:
            nn.init.constant_(self.transformer_gate, 2.0)  # sigmoid(2) ≈ 0.88

    @classmethod
    def from_pretrained(cls, model_path):
        with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
            config = SafeConfig(json.load(f))
        model = cls(config)

        import safetensors.torch
        sd = safetensors.torch.load_file(os.path.join(model_path, "model.safetensors"))
        new_sd = {}
        for k, v in sd.items():
            k = k.strip()
            if k.startswith("model.layers.0."):
                new_sd[k.replace("model.layers.0.", "prelude.0.")] = v
            elif k.startswith("model.layers.1."):
                new_sd[k.replace("model.layers.1.", "prelude.1.")] = v
            elif k.startswith("model.embed_tokens."):
                new_sd[k.replace("model.embed_tokens.", "embed_tokens.")] = v
            elif k.startswith("model.norm."):
                new_sd[k.replace("model.norm.", "norm.")] = v
            elif k.startswith("lm_head."):
                new_sd[k.replace("lm_head.", "output_head.base_head.")] = v

        model.load_state_dict(new_sd, strict=False)
        model.output_head.base_head.weight = model.embed_tokens.weight
        print("✅ Веса загружены. v7.1.1 готова.")
        return model


# ==========================================
# 8. SMOKE TEST
# ==========================================
def smoke_test():
    print("🧪 Smoke test v7.1.1...")

    config = SafeConfig({
        "hidden_size": 768,
        "vocab_size": 32000,
        "num_attention_heads": 12,
        "num_key_value_heads": 4,
        "intermediate_size": 2048,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "attention_bias": False,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "num_prelude_layers": 2,
        "num_coda_layers": 2,
        "num_loops": 4,
        "state_dim": 256,
        "transformer_heads": 12,
        "transformer_kv_heads": 4,
        "transformer_latent": 256,
        "curriculum_phase": 0,
    })

    model = EnergoAIModelV711(config)

    # Test forward
    input_ids = torch.randint(0, 32000, (2, 128))
    logits, diag = model.forward_with_diagnostics(input_ids)

    print(f"  Logits shape: {logits.shape}")
    print(f"  Prelude norm: {diag['prelude_norm']}")
    print(f"  SSM h_norm: {diag['ssm_h_norm']}")
    print(f"  SSM delta: {diag['ssm_delta_mean']}")
    print(f"  SSM W_in rank: {diag['ssm_W_in_rank']}/256")
    print(f"  Transformer gate: {diag['transformer_gate']}")
    print(f"  Coda norm: {diag['coda_norm']}")

    # Test curriculum
    model.set_curriculum_phase(1)
    logits2, diag2 = model.forward_with_diagnostics(input_ids)
    print(f"  Phase 1 gate: {diag2['transformer_gate']}")

    # Count params
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total / 1e6:.1f}M")

    print("✅ Smoke test PASSED")
    return model


if __name__ == "__main__":
    smoke_test()