# slow_track_fixed.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from final_ssm_phase_beta_1.cores.energoai_ssm_v7 import RMSNorm


class SlowTrackMemory(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_slots=32, surprise_threshold=0.1):
        super().__init__()
        self.N = state_dim      # 128 — размерность SSM state
        self.D = hidden_dim     # 256 — размерность hidden space
        self.K = num_slots
        self.surprise_threshold = surprise_threshold

        # Чтение: query из state [B, N], ключи из slots [B, K, D]
        # Проекция N → D для совместимости с слотами
        self.W_q = nn.Linear(self.N, self.D, bias=False)
        self.gate = nn.Linear(self.N, 1, bias=True)

        # Запись: сжимаем state [B, N] → [B, D] для слотов
        self.compress = nn.Linear(self.N, self.D, bias=False)
        self.compress_norm = RMSNorm(self.D, eps=1e-6)

        # Инициализация
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)  # старт ~0.12
        nn.init.xavier_uniform_(self.W_q.weight, gain=0.5)
        nn.init.xavier_uniform_(self.compress.weight, gain=0.5)

    def read(self, h_f, slots):
        """
        h_f: [B, N] — SSM state
        slots: [B, K, D] — memory slots
        """
        Q = self.W_q(h_f).unsqueeze(1)              # [B, 1, D]
        K = slots                                    # [B, K, D]
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.D)  # [B, 1, K]
        attn_weights = F.softmax(scores, dim=-1)     # [B, 1, K]
        r_t = torch.bmm(attn_weights, slots).squeeze(1)  # [B, D]
        g_t = torch.sigmoid(self.gate(h_f))          # [B, 1]
        return r_t, g_t

    def update_slots(self, h_f, slots, h_prev, surprise_threshold=None):
        """
        h_f: [B, N] — текущее SSM state
        slots: [B, K, D]
        h_prev: [B, N] — предыдущее SSM state (None на первом шаге)
        """
        if h_prev is None:
            return slots, h_f.detach()

        # Относительное изменение (не абсолютное!)
        surprise = torch.norm(h_f - h_prev, dim=-1)       # [B]
        prev_norm = h_prev.norm(dim=-1) + 1e-6
        relative_change = surprise / prev_norm            # [B]

        threshold = surprise_threshold if surprise_threshold is not None else self.surprise_threshold
        mask = (relative_change > threshold).float().view(-1, 1, 1)  # [B, 1, 1]

        # Сжимаем state → hidden space для слотов
        compressed = self.compress(h_f)                   # [B, D]
        compressed = self.compress_norm(compressed)

        # FIFO сдвиг
        shifted = torch.roll(slots, shifts=1, dims=1)
        shifted[:, 0, :] = compressed

        # Per-example обновление
        updated_slots = torch.where(mask.bool(), shifted, slots)
        return updated_slots, h_f.detach()

    def forward(self, h_f, slots, h_prev=None):
        """
        Unified: всегда читаем, обновляем если h_prev есть.
        """
        r_t, g_t = self.read(h_f, slots)
        updated_slots, new_h_prev = self.update_slots(h_f, slots, h_prev)
        return r_t, g_t, updated_slots, new_h_prev