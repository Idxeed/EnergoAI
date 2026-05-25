# prelude_v6.py
"""
Prelude v6: Улучшенные блоки трансформера для EnergoAI.
Содержит GatedDifferentialAttention и другие компоненты нового поколения.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    """Вращение половины размерности для RoPE."""
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Применяет RoPE к запросам и ключам."""
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class GatedDifferentialAttention(nn.Module):
    """
    Гибрид Gated Attention (NeurIPS 2025) и Differential Transformer (ICLR 2025).

    Две карты внимания вычитаются, подавляя шум.
    Сигмоидальный гейт на выходе решает, насколько важен сигнал.

    Совместим с SafeConfig (EnergoAI).
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.rope_theta = float(config.rope_theta)
        self.attention_bias = bool(config.attention_bias)

        # Стандартные проекции Q, K, V
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=self.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=self.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=self.attention_bias)

        # Выходная проекция
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=self.attention_bias)

        # Гейт на выходе внимания: Y' = Y * sigmoid(X @ W_gate)
        self.gate_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 2.0)  # bias=2 → sigmoid(2)=0.88, почти открыт

        # Дифференциальный параметр λ (лямбда) — учится подавлять шум
        self.lambda_init = 0.8
        self.lambda_q1 = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        # Инициализация малых случайных значений для дифференциальных весов
        for p in [self.lambda_q1, self.lambda_k1, self.lambda_q2, self.lambda_k2]:
            nn.init.normal_(p, mean=0.0, std=0.1)

        # RoPE
        inv_freq = 1.0 / (
                    self.rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / float(self.head_dim)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _get_cos_sin(self, seq_len, device, dtype):
        """Вычисляет cos и sin для RoPE."""
        t = torch.arange(int(seq_len), device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None, :, :].to(dtype), emb.sin()[None, None, :, :].to(dtype)

    def _differential_attention(self, q1, k1, q2, k2, v, attn_mask=None):
        """
        Дифференциальное внимание: softmax(Q1·K1) - λ·softmax(Q2·K2).
        Разность двух карт внимания подавляет шум, оставляя только значимые связи.
        """
        # Первая карта внимания
        scores1 = torch.matmul(q1, k1.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores1 = scores1.masked_fill(attn_mask == 0, float('-inf'))
        attn1 = F.softmax(scores1, dim=-1)

        # Вторая карта внимания
        scores2 = torch.matmul(q2, k2.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores2 = scores2.masked_fill(attn_mask == 0, float('-inf'))
        attn2 = F.softmax(scores2, dim=-1)

        # Дифференциальный выход: A1 - λ·A2
        lambda_val = torch.exp(self.lambda_q1.mean() + self.lambda_k1.mean() -
                               self.lambda_q2.mean() - self.lambda_k2.mean())
        lambda_val = torch.sigmoid(lambda_val) * self.lambda_init
        diff_attn = attn1 - lambda_val * attn2

        return torch.matmul(diff_attn, v)

    def forward(self, hidden_states, attention_mask=None):
        bsz, q_len = hidden_states.shape[:2]

        # Проекции Q, K, V
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE
        cos, sin = self._get_cos_sin(q_len, hidden_states.device, hidden_states.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Grouped Query Attention (GQA): повторяем KV-головы, если их меньше
        if self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # Дифференциальные компоненты: разбиваем Q и K на Q1, Q2 и K1, K2
        # с помощью обучаемых весов λ
        q1 = q + self.lambda_q1.unsqueeze(0).unsqueeze(2)  # [B, heads, L, D]
        q2 = q + self.lambda_q2.unsqueeze(0).unsqueeze(2)
        k1 = k + self.lambda_k1.unsqueeze(0).unsqueeze(2)
        k2 = k + self.lambda_k2.unsqueeze(0).unsqueeze(2)

        # Маска внимания (каузальная + паддинг)
        if attention_mask is not None:
            attn_mask = attention_mask.bool().unsqueeze(1).unsqueeze(2)
        else:
            attn_mask = None

        # Дифференциальное внимание
        attn_out = self._differential_attention(q1, k1, q2, k2, v, attn_mask)

        # Выходная проекция
        out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        out = self.o_proj(out)

        # Gated Attention: Y' = Y * sigmoid(X @ W_gate)
        gate = torch.sigmoid(self.gate_proj(hidden_states))
        out = out * gate

        return out


class ExpandedGLUMLP(nn.Module):
    """
    Улучшенный MLP с механизмом Gated Linear Unit (GLU) и расширенным intermediate слоем.
    Вместо стандартного FFN (gate + up + down) использует GLU-структуру:
    output = W_down(SiLU(W_gate(x)) * W_up(x)).
    Промежуточный размер (intermediate_size) задаётся с множителем 6x (по умолчанию 4x).
    """

    def __init__(self, config, intermediate_multiplier=6):
        super().__init__()
        hs = int(config.hidden_size)
        ims = int(config.hidden_size * intermediate_multiplier)  # 6x по умолчанию
        bias = bool(config.attention_bias)

        # GLU-проекции
        self.gate_proj = nn.Linear(hs, ims, bias=bias)
        self.up_proj = nn.Linear(hs, ims, bias=bias)
        self.down_proj = nn.Linear(ims, hs, bias=bias)

        # Инициализация с усилением для богатого сигнала
        nn.init.xavier_uniform_(self.gate_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.up_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.down_proj.weight, gain=0.2)
        if bias:
            nn.init.zeros_(self.gate_proj.bias)
            nn.init.zeros_(self.up_proj.bias)
            nn.init.zeros_(self.down_proj.bias)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class EnergoAIDecoderLayerV6(nn.Module):
    """
    Слой декодера нового поколения для EnergoAI v6.
    Включает Gated Differential Attention и Expanded GLU MLP.
    Полностью независим от весов Qwen — девственно чистый Prelude.
    """

    def __init__(self, config, intermediate_multiplier=6):
        super().__init__()
        hs, eps = int(config.hidden_size), float(config.rms_norm_eps)

        # Нормализация перед вниманием
        self.input_layernorm = RMSNorm(hs, eps=eps)
        # Улучшенное внимание (дифференциальное + гейтированное)
        self.self_attn = GatedDifferentialAttention(config)
        # Нормализация перед MLP
        self.post_attention_layernorm = RMSNorm(hs, eps=eps)
        # Усиленный MLP с GLU
        self.mlp = ExpandedGLUMLP(config, intermediate_multiplier=intermediate_multiplier)

    def forward(self, x, mask=None):
        # Блок внимания
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, attention_mask=mask)
        x = residual + x

        # Блок MLP
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x