import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torchdiffeq

from .space_ode import SpaceODE, SpaceODEFunc
from .self_attention import AttentionLayer


class TimeODE(nn.Module):
    def __init__(self,
                 odefunc: nn.Module,
                 method,
                 time_1, step_size_1,
                 rtol, atol,
                 perturb=False):
        super().__init__()

        self.odefunc = odefunc
        self.method = method
        self.time_1 = time_1
        self.step_size_1 = step_size_1
        self.perturb = perturb
        self.atol = atol
        self.rtol = rtol

    def forward(self, X: Tensor) -> Tensor:
        '''X: [B, T, N, C]'''

        self.integration_time = torch.tensor([0, self.time_1]).float().type_as(X)
        out = torchdiffeq.odeint(
            self.odefunc,
            X,
            self.integration_time,
            rtol=self.rtol, atol=self.atol,
            method=self.method,
            options=dict(step_size=self.step_size_1, perturb=self.perturb))

        return out[-1]


class TimeODEFunc(nn.Module):
    def __init__(self, time_ode_net: nn.Module):
        super().__init__()
        self.time_ode_net = time_ode_net
        self.nfe = 0

    def forward(self, t, x: Tensor) -> Tensor:
        '''x: [B, T, N, C]'''

        self.nfe += 1
        x = self.time_ode_net(x)
        return x


class TimeODENet(nn.Module):
    def __init__(self,
                 hidden_channels,
                 feed_forward_dim,
                 num_heads,
                 dropout,
                 method,
                 time_2, step_size_2,
                 rtol, atol,
                 perturb=False):
        super().__init__()
        self.method = method
        self.time_2 = time_2
        self.step_size_2 = step_size_2
        self.rtol = rtol
        self.atol = atol
        self.perturb = perturb
        self.dropout = dropout

        self.filter_att = SelfAttentionLayer(
            model_dim=hidden_channels,
            feed_forward_dim=feed_forward_dim,
            num_heads=num_heads,
            dropout=dropout,
            mask=False
        )

        self.gate_att = SelfAttentionLayer(
            model_dim=hidden_channels,
            feed_forward_dim=feed_forward_dim,
            num_heads=num_heads,
            dropout=dropout,
            mask=False
        )

        self.space_ode = SpaceODE(
            SpaceODEFunc(hidden_channels, num_heads),
            hidden_channels,
            feed_forward_dim,
            dropout,
            method,
            time_2, step_size_2,
            rtol, atol, perturb
        )

    def forward(self, x: Tensor) -> Tensor:
        '''x: [B, T, N, C]'''

        _filter = self.filter_att(x)
        _filter = torch.tanh(_filter)

        _gate = self.gate_att(x)
        _gate = torch.sigmoid(_gate)

        x = _filter * _gate

        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.space_ode(x)

        x = F.layer_norm(x, tuple(x.shape[1:]), weight=None, bias=None, eps=1e-5)

        return x


class SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(1, -2)
        return out
