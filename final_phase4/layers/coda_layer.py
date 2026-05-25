# coda_layer.py
"""
State-Conditioned Coda с Checkpoint Attention для EnergoAI SSM.
"""

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


class CodaBlock(nn.Module):
    """Базовый блок без checkpoint attention."""

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

        mlp_norm = mlp_out.norm(dim=-1, keepdim=True)
        scale = (50.0 / mlp_norm).clamp(max=1.0)
        mlp_out = mlp_out * scale

        return hidden_states + self.res_gate * mlp_out


class CodaBlockWithCheckpoints(CodaBlock):
    """CodaBlock + checkpoint attention."""

    def __init__(self, hidden_dim, intermediate, state_dim=128):
        super().__init__(hidden_dim, intermediate)

        self.checkpoint_q = nn.Linear(hidden_dim, 64)
        self.checkpoint_k = nn.Linear(state_dim, 64)
        self.checkpoint_v = nn.Linear(state_dim, hidden_dim)
        self.checkpoint_gate = nn.Linear(hidden_dim, 1)

        nn.init.xavier_uniform_(self.checkpoint_q.weight, gain=0.1)
        nn.init.xavier_uniform_(self.checkpoint_k.weight, gain=0.1)
        nn.init.xavier_uniform_(self.checkpoint_v.weight, gain=0.01)

    def forward(self, hidden_states, state_bias, checkpoints=None):
        # Базовая часть
        x = self.norm(hidden_states)
        x = x + state_bias
        mlp_out = self.down(F.silu(self.gate(x)) * self.up(x))

        mlp_norm = mlp_out.norm(dim=-1, keepdim=True)
        scale = (50.0 / mlp_norm).clamp(max=1.0)
        mlp_out = mlp_out * scale

        # Checkpoint attention
        if checkpoints is not None and checkpoints.shape[0] > 0:
            num_ckpt, B, _ = checkpoints.shape
            L = hidden_states.shape[1]

            Q = self.checkpoint_q(hidden_states)  # [B, L, 64]
            ckpt_perm = checkpoints.permute(1, 0, 2)  # [B, num_ckpt, state_dim]
            K = self.checkpoint_k(ckpt_perm)  # [B, num_ckpt, 64]
            V = self.checkpoint_v(ckpt_perm)  # [B, num_ckpt, hidden_dim]

            scores = torch.einsum('bld,bnd->bln', Q, K) / 8.0
            attn_weights = F.softmax(scores, dim=-1)
            checkpoint_context = torch.einsum('bln,bnd->bld', attn_weights, V)

            gate = torch.sigmoid(self.checkpoint_gate(hidden_states.mean(dim=1, keepdim=True)))
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
        nn.init.xavier_uniform_(self.state_proj.weight, gain=0.1)

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
# SMOKE TEST
# ==========================================
def smoke_test_coda():
    print("🧪 Coda smoke test...")

    B, L, D, N = 2, 10, 896, 128
    hidden = torch.randn(B, L, D) * 0.01
    h_depth = torch.randn(B, N)

    # С checkpoints
    checkpoints = torch.randn(3, B, N)  # 3 checkpoints

    coda = StateConditionedMLP(D, N, num_layers=2)

    with torch.no_grad():
        out = coda(hidden, h_depth, checkpoints)

    in_norm = hidden.norm(dim=-1).mean().item()
    out_norm = out.norm(dim=-1).mean().item()

    print(f"  Input norm:  {in_norm:.2f}")
    print(f"  Output norm: {out_norm:.2f}")
    print(f"  Res gates:   {coda.diagnostics['res_gates']}")
    print(f"  Layer norms: {coda.diagnostics['coda_layer_norms']}")
    print(f"  Checkpoints: {coda.diagnostics['num_checkpoints_used']}")

    assert out.shape == hidden.shape, "Shape mismatch"
    assert out_norm < 40, f"Coda взрывается: {out_norm}"
    print("✅ Coda smoke test PASSED")


if __name__ == "__main__":
    smoke_test_coda()