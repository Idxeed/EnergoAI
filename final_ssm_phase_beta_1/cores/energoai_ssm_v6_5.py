# energoai_ssm_v6_5.py
"""
EnergoAI SSM v6.5 — фундаментальный фикс коллапса памяти.
- Hard State Constraint (запрет на взрыв нормы h)
- Decoupled Δ и B (независимое управление забыванием и записью)
- Explicit Forget Gate G_t (модель учится забывать осознанно)
- Spectral Clamp на A_bar (стабильность дискретизации)
- Per-loop decay sigmoid(0)=0.5 (между петлями реально забываем)
- Чекпоинты SSM через чанковый scan (быстро + память контекста)
- Causal Differential Attention в Prelude
- Per-token gate в Coda (не размытие по mean)
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
        self.num_loops = raw_dict.get("num_loops", 5)
        self.state_dim = raw_dict.get("state_dim", 128)
        self.checkpoint_every = raw_dict.get("checkpoint_every", 500)

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
# 2. PRELUDE v6.5 (fixed)
# ==========================================
class GatedDifferentialAttention(nn.Module):
    """
    Исправленное внимание:
    - Causal mask (треугольная) обязательна
    - Per-head lambda (не один скаляр на всё)
    - Gate на выходе
    """
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

        # Per-head differential lambda
        self.lambda_param = nn.Parameter(torch.full((self.num_heads,), 0.8))

        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / float(self.head_dim)))
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

        # Causal mask (строго!)
        causal_mask = torch.triu(torch.ones(q_len, q_len, device=hidden_states.device, dtype=torch.bool), diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, q_len, q_len]

        if attention_mask is not None:
            pad_mask = (~attention_mask.bool()).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L] — True где pad
            combined_mask = causal_mask | pad_mask
        else:
            combined_mask = causal_mask

        # Scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(combined_mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)

        # Differential: подавляем фоновую массу внимания через per-head lambda
        lambda_val = torch.sigmoid(self.lambda_param).view(1, self.num_heads, 1, 1)
        attn = attn - lambda_val * attn.mean(dim=-1, keepdim=True)
        attn = torch.clamp(attn, min=0.0)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        out = self.o_proj(out)

        gate = torch.sigmoid(self.gate_proj(hidden_states))
        out = out * gate
        return out


class ExpandedGLUMLP(nn.Module):
    def __init__(self, config, intermediate_multiplier=6):
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


class EnergoAIDecoderLayerV6(nn.Module):
    def __init__(self, config, intermediate_multiplier=6):
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
# 3. SSM CORE v6.5 — фундаментальный фикс
# ==========================================
class FixedSSMCore(nn.Module):
    """
    Исправленное SSM-ядро:
    - Decoupled B (независим от Δ)
    - Explicit Forget Gate G_t
    - Hard State Constraint (clip по норме, не косметика)
    - Spectral clamp на A_bar
    - State Dropout
    """
    def __init__(self, hidden_dim, state_dim=128):
        super().__init__()
        self.D = hidden_dim
        self.N = state_dim

        self.rms_norm = RMSNorm(hidden_dim, eps=1e-6)

        # Проекции
        self.proj_in = nn.Linear(self.D, self.D, bias=False)
        self.proj_DT = nn.Linear(self.D, self.D, bias=False)
        self.proj_B = nn.Linear(self.D, self.N, bias=False)   # Decoupled input
        self.proj_C = nn.Linear(self.D, self.N, bias=False)   # Readout

        # Explicit forget gate
        self.proj_G = nn.Linear(self.D, self.N, bias=True)

        # Depth conditioning для Δ
        self.W_DT = nn.Parameter(torch.randn(self.D, self.N) * 0.02)
        self.W_depth = nn.Parameter(torch.randn(self.D, self.N) * 0.02)

        # Output
        self.W_gate = nn.Linear(self.D, self.D, bias=False)
        self.W_out = nn.Linear(self.N, self.D, bias=False)
        self.D_skip = nn.Parameter(torch.ones(self.D) * 0.05)

        # SSM dynamics
        self.A_log = nn.Parameter(torch.randn(self.N) * 0.1 - 2.5)
        self.b_Δ = nn.Parameter(torch.logspace(-4, -1, self.N))  # широкий базовый спектр

        # Delta scale для стартовой нормы
        self.delta_scale = nn.Parameter(torch.tensor(0.1))

        #input gain параметр для запуска голов SSM
        self.input_gain = nn.Parameter(torch.tensor(0.1))

        # Hard State Constraint
        self.state_tau = nn.Parameter(torch.tensor(8.0))

        # State Dropout
        self.state_dropout_p = 0.05

        # Нормализация y
        self.y_norm = RMSNorm(self.D, eps=1e-6)

        self.diagnostics = {}
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            for lin in [self.proj_B, self.proj_C, self.W_out, self.W_gate]:
                nn.init.xavier_uniform_(lin.weight, gain=0.5)
            for lin in [self.proj_in, self.proj_DT]:
                nn.init.xavier_uniform_(lin.weight, gain=0.3)
            nn.init.zeros_(self.proj_G.bias)          # G ≈ 0.5 на старте
            nn.init.xavier_uniform_(self.proj_G.weight, gain=0.3)
            nn.init.xavier_uniform_(self.W_DT, gain=0.02)
            nn.init.xavier_uniform_(self.W_depth, gain=0.02)
            nn.init.constant_(self.input_gain, 0.1)

    def forward(self, x, h_prev, depth_emb_t=None, mask=None,
                prelude_output=None, return_delta=False):
        B, L, _ = x.shape
        device, dtype = x.device, x.dtype

        if depth_emb_t is None:
            depth_emb_t = torch.zeros(self.D, device=device, dtype=dtype)

        x_norm = self.rms_norm(x)
        x_in = self.proj_in(x_norm)
        x_DT = self.proj_DT(x_norm)
        x_B_raw = self.proj_B(x_norm)   # [B, L, N] — независимый вход
        input_term = self.input_gain * x_B_raw  # масштабируем вход
        x_C_raw = self.proj_C(x_norm)

        x_in = x_in + depth_emb_t.unsqueeze(0).unsqueeze(0)

        # Δ = exp( ... ) с жёстким clamp, чтобы не взорвалось
        dt_linear = x_DT @ self.W_DT + depth_emb_t.unsqueeze(0).unsqueeze(0) @ self.W_depth
        dt_logits = dt_linear + self.b_Δ.view(1, 1, -1)
        dt_logits = torch.clamp(dt_logits, min=-10.0, max=5.0)
        Δ = torch.exp(dt_logits)        # [B, L, N]

        A = -torch.exp(self.A_log)      # [N]

        # Spectral clamp: A_bar ∈ (0, 1]
        exponent = torch.clamp(Δ * A.view(1, 1, -1), min=-10.0, max=0.0)
        A_bar = torch.exp(exponent)     # [B, L, N]

        # Explicit Forget Gate: G≈0.5 на старте
        G = torch.sigmoid(self.proj_G(x_norm))  # [B, L, N]

        # Hard State Constrained Scan
        h = h_prev                      # [B, N]
        h_list = []
        tau = F.softplus(self.state_tau) + 5.0  # минимум 5.0

        for t in range(L):
            # Convex combination: G * decayed_old + (1-G) * new_input
            h_candidate = G[:, t] * (A_bar[:, t] * h) + (1.0 - G[:, t]) * input_term[:, t]

            # Hard constraint: если норма > tau — сжимаем
            h_norm = h_candidate.norm(dim=-1, keepdim=True) + 1e-6
            scale = torch.clamp(tau / h_norm, max=1.0)
            h_new = h_candidate * scale

            # State Dropout (только train)
            if self.training and self.state_dropout_p > 0:
                dropout_mask = torch.bernoulli(
                    torch.ones_like(h_new) * (1 - self.state_dropout_p)
                ) / (1 - self.state_dropout_p)
                h_new = h_new * dropout_mask

            # Padding mask
            if mask is not None:
                h_new = torch.where(mask[:, t].unsqueeze(-1).bool(), h_new, h)

            h = h_new
            h_list.append(h)

        h_seq = torch.stack(h_list, dim=1)  # [B, L, N]

        # Output readout
        y_ssm = x_C_raw * h_seq
        y_decoded = self.W_out(y_ssm)
        y = y_decoded + self.D_skip.view(1, 1, -1) * x_in
        y = self.y_norm(y)

        gate = F.silu(self.W_gate(x_norm))
        delta = self.delta_scale * gate * y

        if return_delta:
            if mask is not None:
                delta = delta * mask.unsqueeze(-1).to(delta.dtype)

            if mask is not None:
                lengths = (mask.sum(dim=1) - 1).clamp(min=0, max=L - 1)
                h_depth_out = h_seq[torch.arange(B, device=device), lengths, :]
            else:
                h_depth_out = h_seq[:, -1, :]

            with torch.no_grad():
                self.diagnostics = {
                    "Δ_mean": Δ.mean().item(),
                    "Δ_min": Δ.min().item(),
                    "Δ_max": Δ.max().item(),
                    "A_bar_mean": A_bar.mean().item(),
                    "G_mean": G.mean().item(),
                    "h_norm_raw": h_seq.norm(dim=-1).mean().item(),
                    "h_norm_clipped": tau.item(),
                    "state_tau": tau.item(),
                    "input_gain": self.input_gain.item(),
                    "input_term_norm": input_term.norm().item(),
                }
            return delta, h_depth_out
        else:
            x_out = x + delta
            if mask is not None:
                x_out = torch.where(mask.unsqueeze(-1), x_out, x)

            if mask is not None:
                lengths = (mask.sum(dim=1) - 1).clamp(min=0, max=L - 1)
                h_depth_out = h_seq[torch.arange(B, device=device), lengths, :]
            else:
                h_depth_out = h_seq[:, -1, :]

            return x_out, h_depth_out


# ==========================================
# 4. CHECKPOINT SSM (чанковый, быстрый)
# ==========================================
class CheckpointSSM(nn.Module):
    def __init__(self, ssm_layer, checkpoint_every=500):
        super().__init__()
        self.ssm = ssm_layer
        self.checkpoint_every = checkpoint_every
        self.checkpoints = []

    def forward(self, x, h_depth_in, **kwargs):
        B, L, _ = x.shape
        self.checkpoints = []

        x_out_list = []
        h_prev = h_depth_in

        for start in range(0, L, self.checkpoint_every):
            end = min(start + self.checkpoint_every, L)
            x_chunk = x[:, start:end, :]

            # Обрезаем тензоры-последовательности в kwargs
            kwargs_chunk = {}
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor) and v.dim() >= 2 and v.shape[1] == L:
                    kwargs_chunk[k] = v[:, start:end]
                else:
                    kwargs_chunk[k] = v

            out_chunk, h_prev = self.ssm(x_chunk, h_prev, **kwargs_chunk)
            x_out_list.append(out_chunk)
            self.checkpoints.append(h_prev.clone().detach())

        x_out = torch.cat(x_out_list, dim=1)
        return x_out, h_prev

    def get_checkpoints(self):
        if len(self.checkpoints) == 0:
            return None
        return torch.stack(self.checkpoints, dim=0)


# ==========================================
# 5. CODA v6.5 (fixed gate)
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

    def forward(self, hidden_states, state_bias, checkpoints=None):
        x = self.norm(hidden_states)
        x = x + state_bias
        mlp_out = self.down(F.silu(self.gate(x)) * self.up(x))

        # Мягкий клип вместо жёсткого 50.0
        mlp_norm = mlp_out.norm(dim=-1, keepdim=True)
        scale = torch.clamp(10.0 / mlp_norm, max=1.0)
        mlp_out = mlp_out * scale

        return hidden_states + self.res_gate * mlp_out


class CodaBlockWithCheckpoints(CodaBlock):
    def __init__(self, hidden_dim, intermediate, state_dim=128):
        super().__init__(hidden_dim, intermediate)

        self.checkpoint_q = nn.Linear(hidden_dim, 64)
        self.checkpoint_k = nn.Linear(state_dim, 64)
        self.checkpoint_v = nn.Linear(state_dim, hidden_dim)
        self.checkpoint_gate = nn.Linear(hidden_dim, 1)  # per-token!

        nn.init.xavier_uniform_(self.checkpoint_q.weight, gain=0.1)
        nn.init.xavier_uniform_(self.checkpoint_k.weight, gain=0.1)
        nn.init.xavier_uniform_(self.checkpoint_v.weight, gain=0.01)

    def forward(self, hidden_states, state_bias, checkpoints=None):
        # Базовая часть
        x = self.norm(hidden_states)
        x = x + state_bias
        mlp_out = self.down(F.silu(self.gate(x)) * self.up(x))

        mlp_norm = mlp_out.norm(dim=-1, keepdim=True)
        scale = torch.clamp(10.0 / mlp_norm, max=1.0)
        mlp_out = mlp_out * scale

        # Checkpoint attention
        if checkpoints is not None and checkpoints.shape[0] > 0:
            num_ckpt, B, _ = checkpoints.shape
            L = hidden_states.shape[1]

            Q = self.checkpoint_q(hidden_states)          # [B, L, 64]
            ckpt_perm = checkpoints.permute(1, 0, 2)        # [B, num_ckpt, state_dim]
            K = self.checkpoint_k(ckpt_perm)              # [B, num_ckpt, 64]
            V = self.checkpoint_v(ckpt_perm)              # [B, num_ckpt, hidden_dim]

            scores = torch.einsum('bld,bnd->bln', Q, K) / 8.0
            attn_weights = F.softmax(scores, dim=-1)
            checkpoint_context = torch.einsum('bln,bnd->bld', attn_weights, V)

            # Per-token gate (fixed: не mean по sequence!)
            gate = torch.sigmoid(self.checkpoint_gate(hidden_states))  # [B, L, 1]
            mlp_out = mlp_out + gate * checkpoint_context

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

        # Первый слой — с checkpoint attention, остальные — базовые
        self.layers = nn.ModuleList([
            CodaBlockWithCheckpoints(hidden_dim, intermediate, state_dim)
        ])
        for _ in range(num_layers - 1):
            self.layers.append(CodaBlock(hidden_dim, intermediate))

        self.diagnostics = {}

    def forward(self, hidden_states, h_depth, checkpoints=None):
        B, L, D = hidden_states.shape
        state_bias = self.state_proj(h_depth).unsqueeze(1)

        layer_norms = []
        for i, layer in enumerate(self.layers):
            ckpt = checkpoints if i == 0 else None
            hidden_states = layer(hidden_states, state_bias, ckpt)
            layer_norms.append(round(hidden_states.norm(dim=-1).mean().item(), 3))

        self.diagnostics = {
            'coda_layer_norms': layer_norms,
            'state_bias_norm': round(state_bias.norm(dim=-1).mean().item(), 3),
            'res_gates': [round(float(layer.res_gate.item()), 3) for layer in self.layers],
            'num_checkpoints_used': checkpoints.shape[0] if checkpoints is not None else 0
        }

        return hidden_states


# ==========================================
# 6. OUTPUT HEAD
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
# 7. ENERGOAI MODEL v6.5
# ==========================================
class EnergoAIModelV65(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        pad_idx = int(config.pad_token_id) if config.pad_token_id is not None else None
        self.embed_tokens = nn.Embedding(int(config.vocab_size), int(config.hidden_size), padding_idx=pad_idx)

        # Prelude v6.5 (fixed causal + differential)
        self.prelude = nn.ModuleList([
            EnergoAIDecoderLayerV6(config) for _ in range(int(config.num_prelude_layers))
        ])

        # SSM core: фундаментально исправленное ядро + чекпоинты
        self.core_block = CheckpointSSM(
            FixedSSMCore(int(config.hidden_size), state_dim=int(config.state_dim)),
            checkpoint_every=int(getattr(config, 'checkpoint_every', 500))
        )
        self.num_loops = int(config.num_loops)

        # Depth embeddings
        self.depth_emb = nn.Parameter(torch.randn(self.num_loops, int(config.hidden_size)) * 0.02)

        # Начальное состояние
        self.h_depth_init = nn.Parameter(torch.zeros(1, int(config.state_dim)))
        with torch.no_grad():
            self.h_depth_init.normal_(0, 0.01)

        # Per-loop decay: sigmoid(0) = 0.5 — реально забываем между петлями!
        self.depth_decay_logits = nn.ParameterList([
            nn.Parameter(torch.zeros(self.N))  # sigmoid(0) = 0.5, градиент = 0.25
            for _ in range(self.num_loops)
        ])

        # Нормализация h_depth после каждого цикла (стандартный weight=1.0)
        self.h_depth_norms = nn.ModuleList([
            RMSNorm(int(config.state_dim), eps=float(config.rms_norm_eps)) for _ in range(self.num_loops)
        ])
        for norm in self.h_depth_norms:
            nn.init.constant_(norm.weight, 1.0)

        # Остаточная связь петель
        self.loop_res_weight = nn.Parameter(torch.tensor(0.5))

        # Bridge
        self.bridge = nn.Sequential(
            RMSNorm(int(config.hidden_size), eps=float(config.rms_norm_eps)),
            nn.Linear(int(config.hidden_size), int(config.hidden_size), bias=False),
            RMSNorm(int(config.hidden_size), eps=float(config.rms_norm_eps)),
        )
        nn.init.xavier_uniform_(self.bridge[1].weight, gain=1.0)

        # Coda
        self.coda = StateConditionedMLP(
            hidden_dim=int(config.hidden_size),
            state_dim=int(config.state_dim),
            intermediate_factor=2.5,
            num_layers=int(config.num_coda_layers)
        )

        self.memory_skip = nn.Linear(int(config.state_dim), int(config.hidden_size), bias=False)
        nn.init.xavier_uniform_(self.memory_skip.weight, gain=0.5)

        self.norm = RMSNorm(int(config.hidden_size), eps=float(config.rms_norm_eps))
        self.output_head = FlexibleOutputHead(config, memory_dim=int(config.state_dim))
        self.output_head.base_head.weight = self.embed_tokens.weight

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

        # Prelude
        for layer in self.prelude:
            hidden_states = layer(hidden_states, attention_mask)
        prelude_norm = hidden_states.norm(dim=-1).mean().item()
        prelude_out = hidden_states

        loop_diagnostics = []
        h_depth = self.h_depth_init.expand(bsz, -1)

        for t in range(self.num_loops):
            depth_emb_t = self.depth_emb[t]

            # SSM в delta-режиме (чанковый scan + чекпоинты)
            delta, h_depth = self.core_block(
                hidden_states, h_depth,
                depth_emb_t=depth_emb_t,
                mask=attention_mask,
                prelude_output=prelude_out,
                return_delta=True
            )

            # Остаточная связь
            hidden_states = hidden_states + self.loop_res_weight * delta

            # Inter-loop decay: ЗАБЫВАЕМ, а не накапливаем
            decay = torch.sigmoid(self.depth_decay_logits[t])  # [N]
            h_depth = h_depth * decay

            # Нормализация
            h_depth = self.h_depth_norms[t](h_depth)

            # Hard constraint между петлями (двойная защита от взрыва)
            h_norm = h_depth.norm(dim=-1, keepdim=True) + 1e-6
            tau_loop = 8.0
            scale = torch.clamp(tau_loop / h_norm, max=1.0)
            h_depth = h_depth * scale

            loop_diag = dict(self.core_block.ssm.diagnostics)
            loop_diag["loop"] = t
            loop_diag["decay"] = decay.mean().item()
            loop_diag["h_depth_norm_after_fix"] = h_depth.norm(dim=-1).mean().item()
            loop_diagnostics.append(loop_diag)

        core_norm = hidden_states.norm(dim=-1).mean().item()

        # Чекпоинты для Coda
        checkpoints = self.core_block.get_checkpoints()

        hidden_states = self.bridge(hidden_states)
        bridge_norm = hidden_states.norm(dim=-1).mean().item()

        # Нормализуем глубинное состояние перед Coda
        h_depth_norm = self.h_depth_norms[-1](h_depth)
        hidden_states = self.coda(hidden_states, h_depth_norm, checkpoints)
        coda_norm = hidden_states.norm(dim=-1).mean().item()

        memory_signal = self.memory_skip(h_depth_norm).unsqueeze(1)  # [B, 1, D]
        hidden_states = hidden_states + memory_signal
        hidden_states = self.norm(hidden_states)

        head_input_norm = hidden_states.norm(dim=-1).mean().item()
        logits = self.output_head(hidden_states, h_depth)

        diagnostics = {
            "prelude_norm": round(prelude_norm, 3),
            "core_norm_after_ssm": round(core_norm, 3),
            "bridge_norm": round(bridge_norm, 3),
            "coda_norm": round(coda_norm, 3),
            "head_input_norm": round(head_input_norm, 3),
            "loop_stats": loop_diagnostics,
            "h_depth_final_norm": round(h_depth.norm(dim=-1).mean().item(), 3),
            "coda_res_gates": self.coda.diagnostics.get('res_gates', []),
            "coda_layer_norms": self.coda.diagnostics.get('coda_layer_norms', []),
            "num_checkpoints": self.coda.diagnostics.get('num_checkpoints_used', 0),
            "loop_res_weight": self.loop_res_weight.item(),
            "depth_decay_mean": torch.stack([torch.sigmoid(d) for d in self.depth_decay_logits]).mean().item()
        }

        return logits.view(bsz, seq_len, -1), diagnostics

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
        print("✅ Веса загружены. Tied weights восстановлены. v6.5 готова.")
        return model