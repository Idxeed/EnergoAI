# ssm_layer_v3.py
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SSMLayer(nn.Module):
    def __init__(self, hidden_dim, state_dim=128):
        super().__init__()
        self.D = hidden_dim
        self.N = state_dim

        self.rms_norm = RMSNorm(hidden_dim, eps=1e-6)

        # Проекции с увеличенным gain
        self.proj_in = nn.Linear(self.D, self.D, bias=False)
        self.proj_DT = nn.Linear(self.D, self.D, bias=False)
        self.proj_B  = nn.Linear(self.D, self.N, bias=False)
        self.proj_C  = nn.Linear(self.D, self.N, bias=False)

        self.W_gate = nn.Linear(self.D, self.D, bias=False)
        self.W_out  = nn.Linear(self.N, self.D, bias=False)

        # Параметры SSM
        self.A_log  = nn.Parameter(torch.zeros(self.N))
        self.D_skip = nn.Parameter(torch.ones(self.D) * 0.05)
        self.b_Δ = nn.Parameter(torch.logspace(-4, -1, self.N))   # softplus ≈ 1

        # Depth conditioning
        self.W_DT   = nn.Parameter(torch.randn(self.D, self.N) * 0.02)
        self.W_depth = nn.Parameter(torch.randn(self.D, self.N) * 0.02)

        # Инъекция Prelude
        self.input_gain = nn.Parameter(torch.tensor(0.3))
        self.input_injection = nn.Linear(self.D, self.N, bias=False)

        # Обучаемый масштаб delta (чтобы стартовая норма была значимой)
        self.delta_scale = nn.Parameter(torch.tensor(0.1))

        # Дополнительная норма для y перед gate
        self.y_norm = RMSNorm(self.D, eps=1e-6)

        self.diagnostics = {}
        self._init_contractive()

    def _init_contractive(self):
        with torch.no_grad():
            self.A_log.normal_(-3.0, 0.1)
            self.D_skip.data.fill_(0.05)

            # Увеличиваем gain для ключевых проекций
            for lin in [self.proj_B, self.proj_C, self.W_out, self.W_gate]:
                nn.init.xavier_uniform_(lin.weight, gain=0.5)
            # Остальные оставляем умеренными
            for lin in [self.proj_in, self.proj_DT]:
                nn.init.xavier_uniform_(lin.weight, gain=0.3)
            nn.init.xavier_uniform_(self.input_injection.weight, gain=0.2)

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

        Δ = torch.exp(
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

        h_seq_list = []
        h_prev = h_depth_in
        for i in range(L):
            h = (A_bar[:, i, :] * h_prev +
                 self.input_gain * B_bar[:, i, :] +
                 injected_seq[:, i, :])
            if mask is not None:
                h = torch.where(mask[:, i].unsqueeze(-1), h, h_prev)
            h_seq_list.append(h)
            h_prev = h
        h_seq = torch.stack(h_seq_list, dim=1)

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
                lengths = (mask.sum(dim=1) - 1).clamp(min=0, max=L-1)
                h_depth_out = h_seq[torch.arange(B, device=device), lengths, :]
            else:
                h_depth_out = h_seq[:, -1, :]

            with torch.no_grad():
                self.diagnostics = {
                    "Δ_mean": Δ.mean().item(),
                    "A_max_eig": float(torch.max(torch.exp(self.A_log))),
                    "h_depth_norm": h_depth_out.norm(dim=-1).mean().item(),
                    "delta_norm": delta.norm(dim=-1).mean().item(),
                    "input_gain": self.input_gain.item(),
                    "delta_scale": self.delta_scale.item(),
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