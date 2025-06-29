import torch
from torch import nn
import torch.nn.functional as F

import torchdiffeq

from .attention import SelfAttentionLayer


class ODEBlock(nn.Module):
    def __init__(self, odefunc, method, step_size, rtol, atol, adjoint=False, perturb=False):
        super().__init__()
        self.odefunc = odefunc
        self.method = method
        self.step_size = step_size
        self.adjoint = adjoint
        self.perturb = perturb
        self.atol = atol
        self.rtol = rtol

    def forward(self, x, t):
        self.integration_time = torch.tensor([0, t]).float().type_as(x)
        if self.adjoint:
            out = torchdiffeq.odeint_adjoint(
                self.odefunc,
                x,
                self.integration_time,
                rtol=self.rtol, atol=self.atol,
                method=self.method,
                options=dict(step_size=self.step_size, perturb=self.perturb))
        else:
            out = torchdiffeq.odeint(
                self.odefunc,
                x,
                self.integration_time,
                rtol=self.rtol, atol=self.atol,
                method=self.method,
                options=dict(step_size=self.step_size, perturb=self.perturb))

        return out[-1]


class ODEFunc(nn.Module):
    def __init__(self, stnet):
        super().__init__()
        self.stnet = stnet
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        x = self.stnet(x)
        return x


class STODE(nn.Module):
    def __init__(self,
                 receptive_field,
                 dilation,
                 hidden_channels, feed_forward_dim, num_heads,
                 dropout,
                 method, time, step_size, rtol, atol, adjoint, perturb=False):
        super().__init__()
        self.method = method
        self.time = time
        self.step_size = step_size
        self.rtol = rtol
        self.atol = atol
        self.adjoint = adjoint
        self.perturb = perturb

        self.receptive_field = receptive_field
        self.intermediate_seq_len = receptive_field
        self.graph = None
        self.dropout = dropout
        self.new_dilation = 1
        self.dilation_factor = dilation
        self.inception_1 = dilated_inception(
            hidden_channels, hidden_channels, dilation_factor=1)
        self.inception_2 = dilated_inception(
            hidden_channels, hidden_channels, dilation_factor=1)

        self.spatial_ode = SpatialODE(
            hidden_channels, feed_forward_dim,
            SpatialTransformerODE(hidden_channels, num_heads),
            method, time, step_size, rtol, atol, perturb, adjoint,
            dropout)

    def forward(self, x):
        x = x[..., -self.intermediate_seq_len:]
        for tconv in self.inception_1.tconv:
            tconv.dilation = (1, self.new_dilation)
        for tconv in self.inception_2.tconv:
            tconv.dilation = (1, self.new_dilation)

        filter = self.inception_1(x)
        filter = torch.tanh(filter)
        gate = self.inception_2(x)
        gate = torch.sigmoid(gate)
        x = filter * gate

        self.new_dilation *= self.dilation_factor
        self.intermediate_seq_len = x.size(3)

        # x: [b, c, n, t]
        x = F.dropout(x, self.dropout, training=self.training)

        x = x.transpose(1, 3)
        x = self.spatial_ode(x)
        x = x.transpose(1, 3)

        x = nn.functional.pad(x, (self.receptive_field - x.size(3), 0))

        return x

    def setIntermediate(self, dilation):
        self.new_dilation = dilation
        self.intermediate_seq_len = self.receptive_field


class SpatialTransformerODE(nn.Module):
    def __init__(self, hidden_channels, num_heads):
        super().__init__()
        self.net = SelfAttentionLayer(hidden_channels, num_heads)
        self.outputs = []

    def forward(self, t, x):
        self.outputs.append(x)
        x = self.net(x, dim=2)
        return x


class SpatialODE(nn.Module):
    def __init__(self,
                 model_dim, feed_forward_dim,
                 net,
                 method, time, step_size, rtol, atol, adjoint, perturb,
                 dropout):
        super().__init__()
        self.model_dim = model_dim
        self.ode_net = net
        self.method = method
        self.time = time
        self.step_size = step_size
        self.adjoint = adjoint
        self.rtol = rtol
        self.atol = atol
        self.perturb = perturb
        self.nfe = int(time / step_size)

        m = (self.nfe + 1)
        self.proj = nn.Sequential(
            nn.Linear(model_dim * m, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim))
        self.dropout = nn.Dropout(p=dropout)
        self.ln = nn.LayerNorm(model_dim)

    def forward(self, x):
        residual = x
        self.integration_time = torch.tensor([0, self.time]).float().type_as(x)
        if self.adjoint:
            x = torchdiffeq.odeint_adjoint(
                self.ode_net,
                x,
                self.integration_time,
                rtol=self.rtol,
                atol=self.atol,
                method=self.method,
                options=dict(step_size=self.step_size, perturb=self.perturb))[-1]
        else:
            x = torchdiffeq.odeint(
                self.ode_net,
                x,
                self.integration_time,
                rtol=self.rtol,
                atol=self.atol,
                method=self.method,
                options=dict(step_size=self.step_size, perturb=self.perturb))[-1]

        self.ode_net.outputs.append(x)
        x = torch.cat(self.ode_net.outputs, dim=-1)
        self.ode_net.outputs = []

        x = self.proj(x)
        x = self.dropout(x)
        x = self.ln(x + residual)

        return x


class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=1):
        super().__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(
                nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))

    def forward(self, input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3):]
        x = torch.cat(x, dim=1)
        return x
