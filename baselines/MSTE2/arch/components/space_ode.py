import torch
from torch import nn, Tensor
import torchdiffeq

from .self_attention import AttentionLayer


class SpaceODE(nn.Module):
    def __init__(self,
                 odefunc: nn.Module,
                 model_dim, feed_forward_dim,
                 dropout,
                 method,
                 time_2, step_size_2,
                 rtol, atol,
                 perturb=False):
        super().__init__()

        self.odefunc = odefunc
        self.method = method
        self.time_2 = time_2
        self.step_size_2 = step_size_2
        self.perturb = perturb
        self.atol = atol
        self.rtol = rtol

        self.nfe = int(time_2 / step_size_2)

        m = (self.nfe + 1)
        self.proj = nn.Sequential(
            nn.Linear(model_dim * m, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim))
        self.dropout = nn.Dropout(p=dropout)
        self.ln = nn.LayerNorm(model_dim)

    def forward(self, x: Tensor) -> Tensor:
        '''x: [B, T, N, C]'''

        residual = x

        self.integration_time = torch.tensor([0, self.time_2]).float().type_as(x)
        x = torchdiffeq.odeint(
            self.odefunc,
            x,
            self.integration_time,
            rtol=self.rtol, atol=self.atol,
            method=self.method,
            options=dict(step_size=self.step_size_2, perturb=self.perturb))[-1]

        self.odefunc.outputs.append(x)
        x = torch.cat(self.odefunc.outputs, dim=-1)
        self.odefunc.outputs = []

        x = self.proj(x)
        x = self.dropout(x)
        x = self.ln(x + residual)

        return x


class SpaceODEFunc(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.space_ode_net = SpaceODENet(model_dim, num_heads)
        self.outputs = []

    def forward(self, t, x: Tensor) -> Tensor:
        '''x: [B, T, N, C]'''

        self.outputs.append(x)
        x = self.space_ode_net(x, dim=2)
        return x


class SpaceODENet(nn.Module):
    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()
        self.attn = AttentionLayer(model_dim, num_heads, mask)

    def forward(self, x, dim=-2):
        '''x: [B, T, N, C]'''

        x = x.transpose(dim, -2)
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)

        out = out.transpose(dim, -2)
        return out
