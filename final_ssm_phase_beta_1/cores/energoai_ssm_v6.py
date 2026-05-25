# energoai_ssm_v6.py


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from final_ssm_phase_beta_1.layers.coda_layer import StateConditionedMLP  # оставляем импорт из внешнего файла


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
        # Дополнительные параметры для масштабирования
        self.num_prelude_layers = raw_dict.get("num_prelude_layers", 2)
        self.num_coda_layers = raw_dict.get("num_coda_layers", 2)
        self.num_loops = raw_dict.get("num_loops", 5)
        self.state_dim = raw_dict.get("state_dim", 128)
        for k in ["hidden_size", "num_attention_heads", "num_key_value_heads", "intermediate_size", "vocab_size"]:
            if hasattr(self, k): setattr(self, k, int(getattr(self, k)))
        for k in ["rope_theta", "rms_norm_eps"]:
            if hasattr(self, k): setattr(self, k, float(getattr(self, k)))


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
# 2. TRANSFORMER BLOCKS (Prelude)
# ==========================================
class EnergoAIAttention(nn.Module):
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

        if attention_mask is not None:
            attn_mask = attention_mask.bool().unsqueeze(1).unsqueeze(2)
        else:
            attn_mask = None

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=True)
        return self.o_proj(out.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size))


class EnergoAIMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hs, ims, bias = int(config.hidden_size), int(config.intermediate_size), bool(config.attention_bias)
        self.gate_proj = nn.Linear(hs, ims, bias=bias)
        self.up_proj = nn.Linear(hs, ims, bias=bias)
        self.down_proj = nn.Linear(ims, hs, bias=bias)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class EnergoAIDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        hs, eps = int(config.hidden_size), float(config.rms_norm_eps)
        self.input_layernorm = RMSNorm(hs, eps=eps)
        self.self_attn = EnergoAIAttention(config)
        self.post_attention_layernorm = RMSNorm(hs, eps=eps)
        self.mlp = EnergoAIMLP(config)

    def forward(self, x, mask=None):
        x = x + self.self_attn(self.input_layernorm(x), mask)
        return x + self.mlp(self.post_attention_layernorm(x))


# ==========================================
# 3. SSM CORE LAYER (v3 – delta mode)
# ==========================================
class SSMLayer(nn.Module):
    def __init__(self, hidden_dim, state_dim=128):
        super().__init__()
        self.D = hidden_dim
        self.N = state_dim

        self.rms_norm = RMSNorm(hidden_dim, eps=1e-6)

        # Проекции с усиленной инициализацией
        self.proj_in = nn.Linear(self.D, self.D, bias=False)
        self.proj_DT = nn.Linear(self.D, self.D, bias=False)
        self.proj_B  = nn.Linear(self.D, self.N, bias=False)
        self.proj_C  = nn.Linear(self.D, self.N, bias=False)

        self.W_gate = nn.Linear(self.D, self.D, bias=False)
        self.W_out  = nn.Linear(self.N, self.D, bias=False)

        self.A_log  = nn.Parameter(torch.zeros(self.N))
        self.D_skip = nn.Parameter(torch.ones(self.D) * 0.05)
        self.b_Δ    = nn.Parameter(torch.full((self.N,), 0.5))

        self.W_DT   = nn.Parameter(torch.randn(self.D, self.N) * 0.02)
        self.W_depth = nn.Parameter(torch.randn(self.D, self.N) * 0.02)

        self.input_gain = nn.Parameter(torch.tensor(0.1))
        self.input_injection = nn.Linear(self.D, self.N, bias=False)
        self.input_gate_logit = nn.Parameter(torch.zeros(self.N))  # per-channel gate, sigmoid(0)=0.5
        self.proj_erase = nn.Linear(self.D, self.N, bias=True)

        self.depth_stop_predictor = nn.Sequential(
            nn.Linear(self.D + self.N, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 0 — продолжать, 1 — остановиться
        )

        # Масштабирование дельты (чтобы стартовая норма была ощутимой)
        self.delta_scale = nn.Parameter(torch.tensor(0.1))
        self.y_norm = RMSNorm(self.D, eps=1e-6)  # стабилизация y
        self.max_depth = 6
        self.diagnostics = {}
        self._init_contractive()

    def _init_contractive(self):
        with torch.no_grad():
            self.A_log.normal_(-3.0, 0.1)
            self.D_skip.data.fill_(0.05)
            # Усиленные gain для ключевых проекций
            for lin in [self.proj_B, self.proj_C, self.W_out, self.W_gate]:
                nn.init.xavier_uniform_(lin.weight, gain=0.5)
            for lin in [self.proj_in, self.proj_DT]:
                nn.init.xavier_uniform_(lin.weight, gain=0.3)
            nn.init.zeros_(self.input_injection.weight)
            nn.init.zeros_(self.input_gate_logit)  # стартовый gate ≈ 0.5 для всех каналов
            # Erase gate: на старте ВЫКЛЮЧЕН (sigmoid(-5) ≈ 0.0067)
            nn.init.zeros_(self.proj_erase.weight)
            nn.init.constant_(self.proj_erase.bias, -5.0)
            for module in self.depth_stop_predictor:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.1)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            nn.init.constant_(self.depth_stop_predictor[-2].bias, -3.0)  # предпоследний слой — Linear

    def forward(self, x, h_depth_in, depth_emb_t=None, mask=None,
                prelude_output=None, return_delta=False):
        B, L, _ = x.shape
        device, dtype = x.device, x.dtype
        if depth_emb_t is None:
            depth_emb_t = torch.zeros(self.D, device=device, dtype=dtype)

        x_norm = self.rms_norm(x)
        x_in   = self.proj_in(x_norm)
        x_DT   = self.proj_DT(x_norm)
        x_B_raw = self.proj_B(x_norm)
        x_C_raw = self.proj_C(x_norm)

        x_in = x_in + depth_emb_t.unsqueeze(0).unsqueeze(0)

        Δ = F.softplus(
            x_DT @ self.W_DT +
            depth_emb_t.unsqueeze(0).unsqueeze(0) @ self.W_depth +
            self.b_Δ
        )

        A = -torch.exp(self.A_log)
        A_bar = torch.exp(Δ * A.unsqueeze(0).unsqueeze(0))
        B_bar = Δ * x_B_raw

        if prelude_output is not None:
            e_norm = self.rms_norm(prelude_output)
        else:
            e_norm = x_norm
        injected_seq = self.input_injection(e_norm)

        B, L, _ = x.shape
        device, dtype = x.device, x.dtype

        # Начальное состояние для каждого токена
        h_state = h_depth_in.unsqueeze(1).expand(-1, L, -1).clone()  # [B, L, N]
        active_mask = torch.ones(B, L, dtype=torch.bool, device=device)

        gate_in = torch.sigmoid(self.input_gate_logit)  # [N] — одинаков для всех токенов и глубин

        # Храним средний erase_gate для диагностики
        erase_gates = []

        for depth in range(self.max_depth):
            if not active_mask.any():
                break

            # Вычисляем stop_prob на текущем состоянии
            stop_input = torch.cat([x_norm, h_state], dim=-1)  # [B, L, D+N]
            stop_prob = self.depth_stop_predictor(stop_input).squeeze(-1)  # [B, L]

            # Токены, которые решили остановиться на этой итерации
            sleep_mask = (stop_prob > 0.5) & active_mask
            active_mask = active_mask & ~sleep_mask

            # Обновляем только активные токены
            if active_mask.any():
                # Для всех токенов считаем кандидата, но применять будем только к активным
                # Используем проекции, которые не зависят от глубины
                # A_bar, B_bar, injected_seq имеют размер [B, L, N]
                input_term = self.input_gain * B_bar + injected_seq  # [B, L, N]
                h_candidate = A_bar * h_state + gate_in.unsqueeze(0).unsqueeze(0) * input_term

                # Erase gate (зависит от x_norm, одинаков для всех глубин)
                gate_erase = torch.sigmoid(self.proj_erase(x_norm))  # [B, L, N]
                h_candidate = h_candidate * (1 - gate_erase)
                if depth == 0:  # собираем статистику только для первого прохода
                    erase_gates.append(gate_erase)

                # Применяем только к активным
                h_state = torch.where(active_mask.unsqueeze(-1), h_candidate, h_state)

            # Если маска задана, возвращаем исходное состояние для неактивных токенов
            if mask is not None:
                h_state = torch.where(mask.unsqueeze(-1).bool(), h_state, h_depth_in.unsqueeze(1).expand(-1, L, -1))

        # После всех микро-циклов h_state — финальное состояние
        h_seq = h_state  # [B, L, N] — используется дальше в y_ssm и т.д.

        # Усреднённый erase_gate для диагностики (как раньше)
        if erase_gates:
            erase_gate_mean = torch.stack(erase_gates).mean().item()
        else:
            erase_gate_mean = 0.0
        stop_input = torch.cat([x_norm, h_seq], dim=-1)  # [B, L, D+N]
        stop_prob = self.depth_stop_predictor(stop_input)  # [B, L, 1]
        stop_prob_mean = stop_prob.mean().item()
        y_ssm = x_C_raw * h_seq
        y_decoded = self.W_out(y_ssm)
        y = y_decoded + self.D_skip.unsqueeze(0).unsqueeze(0) * x_in
        y = self.y_norm(y)                # нормализуем y

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
                erase_mean = torch.stack(erase_gates).mean().item() if erase_gates else 0.0
                self.diagnostics = {
                    "Δ_mean": Δ.mean().item(),
                    "Δ_per_channel_mean": Δ.mean(dim=(0, 1)).tolist(),
                    "A_max_eig": float(torch.max(torch.exp(self.A_log))),
                    "h_depth_norm": h_depth_out.norm(dim=-1).mean().item(),
                    "delta_norm": delta.norm(dim=-1).mean().item(),
                    "gate_norm": gate.norm(dim=-1).mean().item(),
                    "y_norm": y.norm(dim=-1).mean().item(),
                    "input_gain": self.input_gain.item(),
                    "delta_scale": self.delta_scale.item(),
                    "erase_gate_mean": erase_mean,
                    "stop_prob_mean": stop_prob_mean
                }
            return delta, h_depth_out
        else:
            x_out = x + delta
            if mask is not None:
                x_out = torch.where(mask.unsqueeze(-1), x_out, x)
            if mask is not None:
                lengths = (mask.sum(dim=1) - 1).clamp(min=0, max=L-1)
                h_depth_out = h_seq[torch.arange(B, device=device), lengths, :]
            else:
                h_depth_out = h_seq[:, -1, :]

            with torch.no_grad():
                self.diagnostics = {
                    "Δ_mean": Δ.mean().item(),
                    "A_max_eig": float(torch.max(torch.exp(self.A_log))),
                    "h_depth_norm": h_depth_out.norm(dim=-1).mean().item(),
                    "gate_norm": gate.norm(dim=-1).mean().item(),
                }
            return x_out, h_depth_out


# ==========================================
# 4. CHECKPOINT SSM WRAPPER
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

        for i in range(L):
            x_t = x[:, i:i + 1, :]                     # [B, 1, D]

            kwargs_t = {k: v for k, v in kwargs.items()}
            # Обрезаем тензоры, зависящие от длины последовательности
            if 'mask' in kwargs_t and kwargs_t['mask'] is not None:
                kwargs_t['mask'] = kwargs_t['mask'][:, i:i + 1]
            if 'prelude_output' in kwargs_t and kwargs_t['prelude_output'] is not None:
                kwargs_t['prelude_output'] = kwargs_t['prelude_output'][:, i:i + 1, :]

            out_t, h_prev = self.ssm(x_t, h_prev, **kwargs_t)
            x_out_list.append(out_t)

            if (i + 1) % self.checkpoint_every == 0:
                self.checkpoints.append(h_prev.clone().detach())

        if L % self.checkpoint_every != 0:
            self.checkpoints.append(h_prev.clone().detach())

        x_out = torch.cat(x_out_list, dim=1)      # [B, L, D]
        return x_out, h_prev

    def get_checkpoints(self):
        if len(self.checkpoints) == 0:
            return None
        return torch.stack(self.checkpoints, dim=0)


# ==========================================
# 5. FLEXIBLE OUTPUT HEAD
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
                    if layer.bias is not None: layer.bias.zero_()

    def forward(self, hidden_states, memory_state):
        bsz, seq_len = hidden_states.shape[:2]
        base = self.base_head(hidden_states)
        mem_exp = memory_state.unsqueeze(1).expand(-1, seq_len, -1)
        delta = self.adapter(torch.cat([hidden_states, mem_exp], dim=-1))
        gate = torch.sigmoid(self.gate_proj(memory_state)).view(bsz, 1, 1)
        return base + gate * delta


# ==========================================
# 6. ENERGOAI MODEL v3
# ==========================================
class EnergoAIModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        pad_idx = int(config.pad_token_id) if config.pad_token_id is not None else None
        self.embed_tokens = nn.Embedding(int(config.vocab_size), int(config.hidden_size), padding_idx=pad_idx)

        # Prelude (2 слоя трансформера)
        self.prelude = nn.ModuleList([EnergoAIDecoderLayer(config) for _ in range(int(config.num_prelude_layers))])

        # SSM ядро с чекпойнтами
        self.core_block = CheckpointSSM(
            SSMLayer(int(config.hidden_size), state_dim=int(config.state_dim)),
            checkpoint_every=500
        )
        self.num_loops = int(config.num_loops)

        # Depth embeddings
        self.depth_emb = nn.Parameter(torch.randn(self.num_loops, int(config.hidden_size)) * 0.02)

        # Начальное глубинное состояние
        self.h_depth_init = nn.Parameter(torch.zeros(1, int(config.state_dim)))
        with torch.no_grad(): self.h_depth_init.normal_(0, 0.01)

        # Обучаемый decay состояния
        self.depth_decay_logit = nn.Parameter(torch.ones(int(config.state_dim)) * 10.0)  # per-channel

        # Нормализация h_depth после каждого цикла
        self.h_depth_norms = nn.ModuleList([
            RMSNorm(int(config.state_dim), eps=float(config.rms_norm_eps)) for _ in range(self.num_loops)
        ])
        for norm in self.h_depth_norms:
            nn.init.constant_(norm.weight, 0.1)

        # Обучаемый вес остаточной связи для петель
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

            # SSM в delta-режиме
            delta, h_depth = self.core_block(
                hidden_states, h_depth,
                depth_emb_t=depth_emb_t,
                mask=attention_mask,
                prelude_output=prelude_out,
                return_delta=True
            )

            # Остаточная связь с обучаемым весом
            hidden_states = hidden_states + self.loop_res_weight * delta

            # Затухание и нормализация состояния
            decay = torch.sigmoid(self.depth_decay_logit)
            h_depth = h_depth * decay
            h_depth = self.h_depth_norms[t](h_depth)

            loop_diag = dict(self.core_block.ssm.diagnostics)
            loop_diag["loop"] = t
            loop_diag["decay"] = decay.mean().item()
            loop_diag["h_depth_norm_after_fix"] = h_depth.norm(dim=-1).mean().item()
            loop_diagnostics.append(loop_diag)

        core_norm = hidden_states.norm(dim=-1).mean().item()

        # Сбор checkpoints для Coda
        checkpoints = self.core_block.get_checkpoints()

        hidden_states = self.bridge(hidden_states)
        bridge_norm = hidden_states.norm(dim=-1).mean().item()

        # Нормализуем глубинное состояние перед Coda
        h_depth_norm = self.h_depth_norms[-1](h_depth)  # используем последний RMSNorm
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
            "depth_decay": torch.sigmoid(self.depth_decay_logit).mean().item()
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
        print("✅ Веса загружены. Tied weights восстановлены. v3 готова.")
        return model