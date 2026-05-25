"""
EnergoAI Lab - Micro Edition (~46M params with default config)
============================================================
Скейл-даун для быстрого прототипирования и тестирования.
Сохраняет всю архитектуру: Prelude + Efficient + Coda + DiffAttn + Latent KV.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ------------------------------------------------------------
# 0. Конфигурация
# ------------------------------------------------------------
@dataclass
class EnergoAIConfig:
    vocab_size: int = 32000
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    num_kv_heads: int = 2
    head_dim: Optional[int] = None

    latent_dim: int = 64
    state_dim: int = 64
    coda_interval: int = 4
    prelude_positions: Tuple[int, ...] = (-2, -1)

    rope_theta: float = 10000.0
    max_seq_len: int = 2048

    mlp_intermediate_factor: int = 4
    efficient_intermediate_factor: int = 3

    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    use_flash_attention: bool = True
    initializer_range: float = 0.02
    gradient_checkpointing: bool = False

    def __post_init__(self):
        if self.head_dim is None:
            assert self.hidden_size % self.num_heads == 0
            self.head_dim = self.hidden_size // self.num_heads
        object.__setattr__(self, "prelude_positions",
            tuple(p if p >= 0 else self.num_layers + p for p in self.prelude_positions))


# ------------------------------------------------------------
# 1. Базовые компоненты
# ------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0, max_seq_len: int = 8192):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("emb", emb, persistent=False)

    def forward(self, position_ids: torch.Tensor, device: torch.device, dtype: torch.dtype):
        emb = self.emb[position_ids]
        if emb.device != device or emb.dtype != dtype:
            emb = emb.to(device=device, dtype=dtype)
        cos = emb.cos().unsqueeze(1)
        sin = emb.sin().unsqueeze(1)
        return cos, sin


def build_padding_mask(attention_mask: Optional[torch.Tensor], dtype: torch.dtype) -> Optional[torch.Tensor]:
    """
        Возвращает маску [B, 1, 1, L] для SDPA.
        0.0 для valid токенов, -inf для padding.
        """
    if attention_mask is None:
        return None
    # [B, L] -> [B, 1, 1, L] для корректного броадкаста в attention
    mask = (1.0 - attention_mask.to(dtype)) * torch.finfo(dtype).min
    return mask.unsqueeze(1).unsqueeze(2)


def align_attention_mask(attn_mask: Optional[torch.Tensor], target_len: int) -> Optional[torch.Tensor]:
    if attn_mask is None or attn_mask.shape[-1] == target_len:
        return attn_mask

    mask_len = attn_mask.shape[-1]
    if mask_len > target_len:
        return attn_mask[..., -target_len:]

    pad_shape = (*attn_mask.shape[:-1], target_len - mask_len)
    pad = attn_mask.new_zeros(pad_shape)
    return torch.cat([pad, attn_mask], dim=-1)


def build_causal_allowed_mask(q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
    past_len = max(k_len - q_len, 0)
    q_pos = torch.arange(q_len, device=device).unsqueeze(1) + past_len
    k_pos = torch.arange(k_len, device=device).unsqueeze(0)
    return k_pos <= q_pos


class FlashAttention(nn.Module):
    def __init__(self, use_flash: bool = True):
        super().__init__()
        self.use_flash = use_flash

    def forward(self, q, k, v, attn_mask: Optional[torch.Tensor] = None, causal: bool = True):
        Lq, Lk = q.shape[2], k.shape[2]
        attn_mask = align_attention_mask(attn_mask, Lk)

        if self.use_flash:
            if causal and (attn_mask is not None or Lq != Lk):
                causal_mask = build_causal_allowed_mask(Lq, Lk, q.device)
                if attn_mask is None:
                    combined = causal_mask.unsqueeze(0).unsqueeze(0)
                else:
                    valid_mask = attn_mask > torch.finfo(attn_mask.dtype).min / 2
                    combined = causal_mask.unsqueeze(0).unsqueeze(0) & valid_mask.expand(-1, -1, Lq, -1)

                float_mask = torch.zeros(combined.shape, device=q.device, dtype=q.dtype)
                float_mask = float_mask.masked_fill(~combined, torch.finfo(q.dtype).min)
                return F.scaled_dot_product_attention(q, k, v, attn_mask=float_mask, is_causal=False)

            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=causal)
        else:
            # Fallback naive attention
            scale = 1.0 / math.sqrt(q.shape[-1])
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if causal:
                causal_mask = build_causal_allowed_mask(Lq, Lk, q.device)
                scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), torch.finfo(scores.dtype).min)
            if attn_mask is not None:
                scores = scores + attn_mask
            attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            return torch.matmul(attn, v)

# ------------------------------------------------------------
# 2. Adaptive Residual Gate
# ------------------------------------------------------------
class AdaptiveResidualGate(nn.Module):
    def __init__(self, init_value: float = -2.0):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(init_value))

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return residual + torch.sigmoid(self.gate) * x


# ------------------------------------------------------------
# 3. Prelude — Differential Attention v7
# ------------------------------------------------------------
class GatedDifferentialAttention(nn.Module):
    def __init__(self, config: EnergoAIConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.attention_bias = config.attention_bias

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=self.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=self.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=self.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=self.attention_bias)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rope = RotaryEmbedding(self.head_dim, config.rope_theta, config.max_seq_len)
        self.attn = FlashAttention(config.use_flash_attention)

        self.lambda_init = 0.8
        self.lambda_scale = nn.Parameter(torch.tensor(0.0))
        self.lambda_q1 = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))

        self.gate_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 2.0)
        self._init_weights()

    def _init_weights(self):
        for m, g in [(self.q_proj, 0.5), (self.k_proj, 0.5), (self.v_proj, 0.5), (self.o_proj, 0.3)]:
            nn.init.xavier_uniform_(m.weight, gain=g)
        if self.attention_bias:
            for m in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
                nn.init.zeros_(m.bias)
        for p in [self.lambda_q1, self.lambda_k1, self.lambda_q2, self.lambda_k2]:
            nn.init.normal_(p, mean=0.0, std=0.1)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False):
        bsz, q_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q.transpose(1, 2)).transpose(1, 2)
        k = self.k_norm(k.transpose(1, 2)).transpose(1, 2)

        cos, sin = self.rope(position_ids, hidden_states.device, hidden_states.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        past_kv = (k, v) if use_cache else None

        attn_mask = build_padding_mask(attention_mask, q.dtype)

        if self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        q1 = q + self.lambda_q1.unsqueeze(0).unsqueeze(2)
        q2 = q + self.lambda_q2.unsqueeze(0).unsqueeze(2)
        k1 = k + self.lambda_k1.unsqueeze(0).unsqueeze(2)
        k2 = k + self.lambda_k2.unsqueeze(0).unsqueeze(2)

        out1 = self.attn(q1, k1, v, attn_mask=attn_mask, causal=True)
        out2 = self.attn(q2, k2, v, attn_mask=attn_mask, causal=True)

        lambda_val = torch.sigmoid(self.lambda_scale) * self.lambda_init
        attn_out = out1 - lambda_val * out2

        out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        out = self.o_proj(out)
        gate = torch.sigmoid(self.gate_proj(hidden_states))
        return out * gate, past_kv


class PreludeGLUMLP(nn.Module):
    def __init__(self, config: EnergoAIConfig):
        super().__init__()
        hidden = config.hidden_size
        inter = hidden * config.mlp_intermediate_factor
        bias = config.attention_bias
        self.gate_proj = nn.Linear(hidden, inter, bias=bias)
        self.up_proj = nn.Linear(hidden, inter, bias=bias)
        self.down_proj = nn.Linear(inter, hidden, bias=bias)
        for m, g in [(self.gate_proj, 0.5), (self.up_proj, 0.5), (self.down_proj, 0.2)]:
            nn.init.xavier_uniform_(m.weight, gain=g)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class PreludeLayer(nn.Module):
    def __init__(self, config: EnergoAIConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GatedDifferentialAttention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = PreludeGLUMLP(config)
        self.res_gate_attn = AdaptiveResidualGate(-2.0)
        self.res_gate_mlp = AdaptiveResidualGate(-2.0)

    def forward(self, x, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False):
        residual = x
        x = self.input_layernorm(x)
        attn_out, past_kv = self.self_attn(x, attention_mask, position_ids, past_key_value, use_cache)
        x = self.res_gate_attn(attn_out, residual)

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = self.res_gate_mlp(x, residual)
        return x, past_kv


# ------------------------------------------------------------
# 4. Efficient — Latent KV + SwiGLU
# ------------------------------------------------------------
class LatentGQAAttention(nn.Module):
    def __init__(self, config: EnergoAIConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.latent_dim = config.latent_dim

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.kv_down = nn.Linear(self.hidden_size, self.latent_dim, bias=False)
        self.k_up = nn.Linear(self.latent_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_up = nn.Linear(self.latent_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rope = RotaryEmbedding(self.head_dim, config.rope_theta, config.max_seq_len)
        self.attn = FlashAttention(config.use_flash_attention)
        for m, g in [(self.q_proj, 0.5), (self.kv_down, 0.5), (self.k_up, 0.5),
                     (self.v_up, 0.5), (self.o_proj, 0.3)]:
            nn.init.xavier_uniform_(m.weight, gain=g)

    def forward(self, x, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False):
        bsz, q_len, _ = x.shape

        q = self.q_proj(x).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        c_kv = self.kv_down(x)
        k = self.k_up(c_kv).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_up(c_kv).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q.transpose(1, 2)).transpose(1, 2)
        k = self.k_norm(k.transpose(1, 2)).transpose(1, 2)

        cos, sin = self.rope(position_ids, x.device, x.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        past_kv = (k, v) if use_cache else None

        attn_mask = build_padding_mask(attention_mask, q.dtype)

        if self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        out = self.attn(q, k, v, attn_mask=attn_mask, causal=True)
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        return self.o_proj(out), past_kv


class SwiGLUFFN(nn.Module):
    def __init__(self, config: EnergoAIConfig):
        super().__init__()
        hidden = config.hidden_size
        inter = hidden * config.efficient_intermediate_factor
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)
        self.res_gate = AdaptiveResidualGate(-2.0)
        for m, g in [(self.gate_proj, 0.5), (self.up_proj, 0.5), (self.down_proj, 0.2)]:
            nn.init.xavier_uniform_(m.weight, gain=g)

    def forward(self, x):
        mlp = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return self.res_gate(mlp, x)


class EfficientLayer(nn.Module):
    def __init__(self, config: EnergoAIConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = LatentGQAAttention(config)
        self.mlp = SwiGLUFFN(config)

    def forward(self, x, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False):
        attn_out, past_kv = self.attn(self.attn_norm(x), attention_mask, position_ids, past_key_value, use_cache)
        x = x + attn_out
        x = self.mlp(x)
        return x, past_kv


# ------------------------------------------------------------
# 5. Coda — State-Conditioned MLP v3
# ------------------------------------------------------------
class CodaBlock(nn.Module):
    def __init__(self, hidden_size: int, state_dim: int, intermediate_factor: float = 2.5, max_checkpoints: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_checkpoints = max_checkpoints
        inter = int(hidden_size * intermediate_factor)

        self.norm = RMSNorm(hidden_size)
        self.gate_proj = nn.Linear(hidden_size, inter, bias=False)
        self.up_proj = nn.Linear(hidden_size, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden_size, bias=False)
        self.res_gate = AdaptiveResidualGate(-2.0)

        self.state_proj = nn.Linear(state_dim, hidden_size, bias=False)
        nn.init.xavier_uniform_(self.state_proj.weight, gain=1.5)

        self.ckpt_q = nn.Linear(hidden_size, 64, bias=False)
        self.ckpt_k = nn.Linear(state_dim, 64, bias=False)
        self.ckpt_v = nn.Linear(state_dim, hidden_size, bias=False)
        self.ckpt_gate = nn.Parameter(torch.tensor(0.0))

        for m, g in [(self.gate_proj, 0.1), (self.up_proj, 0.1), (self.down_proj, 0.01),
                     (self.ckpt_q, 0.1), (self.ckpt_k, 0.1), (self.ckpt_v, 0.01)]:
            nn.init.xavier_uniform_(m.weight, gain=g)

    def forward(self, hidden_states: torch.Tensor,
                h_state: torch.Tensor,
                checkpoints: Optional[torch.Tensor] = None):
        state_bias = self.state_proj(h_state).unsqueeze(1)
        x = self.norm(hidden_states) + state_bias

        mlp_out = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

        if checkpoints is not None and checkpoints.shape[0] > 0:
            ckpt = checkpoints.transpose(0, 1)
            Q = self.ckpt_q(hidden_states)
            K_ckpt = self.ckpt_k(ckpt)
            V_ckpt = self.ckpt_v(ckpt)

            scores = torch.matmul(Q, K_ckpt.transpose(-2, -1)) / math.sqrt(64)
            attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(hidden_states.dtype)
            ckpt_context = torch.matmul(attn_weights, V_ckpt)

            gate = torch.sigmoid(self.ckpt_gate)
            mlp_out = mlp_out + gate * ckpt_context

        return self.res_gate(mlp_out, hidden_states)


class StateConditionedMLP(nn.Module):
    def __init__(self, hidden_dim: int, state_dim: int, intermediate_factor: float = 2.5, num_layers: int = 1):
        super().__init__()
        self.layers = nn.ModuleList([
            CodaBlock(hidden_dim, state_dim, intermediate_factor, max_checkpoints=3)
        ])
        for _ in range(num_layers - 1):
            self.layers.append(CodaBlock(hidden_dim, state_dim, intermediate_factor, max_checkpoints=0))

    def forward(self, hidden_states, h_state, checkpoints=None):
        for i, layer in enumerate(self.layers):
            ckpt = checkpoints if (i == 0 and checkpoints is not None) else None
            hidden_states = layer(hidden_states, h_state, ckpt)
        return hidden_states


# ------------------------------------------------------------
# 6. Полная модель EnergoAI Micro
# ------------------------------------------------------------
class EnergoAI(nn.Module):
    def __init__(self, config: EnergoAIConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList()
        for i in range(config.num_layers):
            if i in config.prelude_positions:
                self.layers.append(PreludeLayer(config, i))
            else:
                self.layers.append(EfficientLayer(config))

        num_codas = config.num_layers // config.coda_interval
        self.coda_blocks = nn.ModuleList()
        self.coda_state_projs = nn.ModuleList()
        for _ in range(num_codas):
            self.coda_blocks.append(StateConditionedMLP(
                config.hidden_size, config.state_dim,
                intermediate_factor=2.5, num_layers=1
            ))
            self.coda_state_projs.append(nn.Linear(config.hidden_size, config.state_dim, bias=False))

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

        self.gradient_checkpointing = config.gradient_checkpointing
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() < 2: continue
            if "self_attn.gate_proj" in name:
                continue
            if "embed_tokens" in name or "lm_head" in name:
                nn.init.normal_(p, mean=0.0, std=self.config.initializer_range)
            elif "state_proj" in name:
                nn.init.normal_(p, mean=0.0, std=self.config.initializer_range * 1.5)
            elif "o_proj" in name or "down_proj" in name:
                nn.init.xavier_uniform_(p, gain=0.3)
            elif any(k in name for k in ("gate_proj", "up_proj", "q_proj", "k_proj", "v_proj", "kv_down", "k_up", "v_up")):
                nn.init.xavier_uniform_(p, gain=0.5)
            else:
                nn.init.xavier_uniform_(p, gain=1.0)

        for name, p in self.named_parameters():
            if p.dim() < 2:
                if "self_attn.gate_proj.bias" in name:
                    nn.init.constant_(p, 2.0)
                elif "bias" in name:
                    nn.init.zeros_(p)
                elif "gate" in name and "gate_proj" not in name and "res_gate" not in name:
                    nn.init.constant_(p, 0.0)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[List] = None,
                use_cache: bool = False,
                return_states: bool = False):
        if self.training:
            use_cache = False

        bsz, seq_len = input_ids.shape
        x = self.embed_tokens(input_ids)

        if past_key_values is None:
            past_key_values = [None] * self.config.num_layers
        elif len(past_key_values) != self.config.num_layers:
            raise ValueError(f"Expected {self.config.num_layers} past_key_values, got {len(past_key_values)}")

        past_seen_tokens = 0
        for past_kv in past_key_values:
            if past_kv is not None:
                past_seen_tokens = past_kv[0].shape[2]
                break

        if attention_mask is None:
            mask_len = past_seen_tokens + seq_len
            attention_mask = torch.ones(bsz, mask_len, device=input_ids.device, dtype=input_ids.dtype)
        if position_ids is None:
            if past_seen_tokens > 0 and attention_mask.shape[-1] == seq_len:
                position_ids = torch.arange(
                    past_seen_tokens,
                    past_seen_tokens + seq_len,
                    device=input_ids.device,
                    dtype=torch.long,
                ).unsqueeze(0).expand(bsz, -1)
                position_ids = position_ids.masked_fill(attention_mask == 0, 0)
            else:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 0)
                position_ids = position_ids[:, -seq_len:]
        elif position_ids.shape[-1] != seq_len:
            position_ids = position_ids[:, -seq_len:]

        next_past_key_values = [] if use_cache else None

        coda_idx = 0
        states_history = []
        all_states = [] if return_states else None

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i]

            if self.gradient_checkpointing and self.training:
                x, new_kv = checkpoint(layer, x, attention_mask, position_ids, past_kv, use_cache, use_reentrant=False)
            else:
                x, new_kv = layer(x, attention_mask, position_ids, past_kv, use_cache)

            if use_cache:
                next_past_key_values.append(new_kv)
            if return_states:
                all_states.append(x.detach().clone())

            if (i + 1) % self.config.coda_interval == 0 and coda_idx < len(self.coda_blocks):
                if past_seen_tokens > 0 or x.shape[1] == 1:
                    h_state = self.coda_state_projs[coda_idx](x[:, -1, :])
                else:
                    seq_mask = attention_mask[:, -x.shape[1]:]
                    seq_lengths = seq_mask.sum(dim=1) - 1
                    seq_lengths = seq_lengths.clamp(min=0, max=x.shape[1] - 1)
                    h_state = self.coda_state_projs[coda_idx](x[torch.arange(bsz, device=x.device), seq_lengths])

                checkpoints = None
                if len(states_history) > 0:
                    ck = states_history[-self.coda_blocks[coda_idx].layers[0].max_checkpoints:]
                    if len(ck) > 0:
                        checkpoints = torch.stack(ck, dim=0)
                x = self.coda_blocks[coda_idx](x, h_state, checkpoints)
                states_history.append(h_state)
                coda_idx += 1

        x = self.norm(x)
        logits = self.lm_head(x)

        output = (logits,)
        if use_cache:
            output += (next_past_key_values,)
        if return_states:
            output += (all_states,)
        return output if len(output) > 1 else logits


# ------------------------------------------------------------
# 7. Smoke Test
# ------------------------------------------------------------
if __name__ == "__main__":
    # --- Micro config: ~46M params with the default 32k vocabulary ---
    cfg = EnergoAIConfig(
        vocab_size=32000,
        hidden_size=512,
        num_layers=8,
        num_heads=8,
        num_kv_heads=2,
        latent_dim=64,
        state_dim=64,
        coda_interval=4,
        prelude_positions=(-2, -1),
        max_seq_len=2048,
        mlp_intermediate_factor=4,
        efficient_intermediate_factor=3,
        gradient_checkpointing=False,
    )

    model = EnergoAI(cfg)
    total = sum(p.numel() for p in model.parameters())
    print(f"[INFO] EnergoAI Micro - {total:,} params ({total/1e6:.2f}M)")

    x = torch.randint(0, 32000, (2, 128))
    mask = torch.ones(2, 128, dtype=torch.long)
    mask[0, 100:] = 0

    # Forward с padding
    with torch.no_grad():
        out = model(x, attention_mask=mask)
        logits = out[0] if isinstance(out, tuple) else out
    print(f"[OK] Forward: {x.shape} -> {logits.shape} | mean={logits.mean().item():.4f}, std={logits.std().item():.4f}")

    # Autoregressive generation
    model.eval()
    with torch.no_grad():
        prompt = x[:, :32]
        prompt_mask = mask[:, :32]
        out, past_kv = model(prompt, attention_mask=prompt_mask, use_cache=True)

        generated = prompt.clone()
        gen_mask = prompt_mask.clone()

        for step in range(5):
            next_token = out[:, -1].argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)
            gen_mask = torch.cat([gen_mask, torch.ones_like(next_token)], dim=1)

            # 🔥 SAFETY: гарантируем, что длина маски == длина кэша + 1 (новый токен)
            kv_len = past_kv[0][0].shape[2] if past_kv is not None else 0
            target_len = kv_len + 1
            if gen_mask.shape[1] != target_len:
                gen_mask = gen_mask[:, :target_len]  # обрезаем, если рассинхрон

            out, past_kv = model(
                next_token,
                attention_mask=gen_mask,
                past_key_values=past_kv,
                use_cache=True
            )
    print(f"[OK] Autoregressive: {prompt.shape} -> {generated.shape} (5 steps)")
    # Gradient checkpointing
    model.gradient_checkpointing = True
    model.train()
    out = model(x, attention_mask=mask)
    loss = out[0].mean()
    loss.backward()
    print("[OK] Gradient checkpointing + backward passed")

    print("\n[OK] Micro model ready for rapid experimentation!")
