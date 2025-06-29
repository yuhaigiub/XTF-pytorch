from .layer import dilated_inception, mixprop, CGP, graph_constructor
import torchdiffeq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class ODEFunc(nn.Module):
    def __init__(self, stnet):
        super(ODEFunc, self).__init__()
        self.stnet = stnet
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        x = self.stnet(x)
        return x


class ODEBlock(nn.Module):
    def __init__(self, odefunc, method, step_size, rtol, atol, adjoint=False, perturb=False):
        super(ODEBlock, self).__init__()
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
            out = torchdiffeq.odeint_adjoint(self.odefunc, x, self.integration_time, rtol=self.rtol, atol=self.atol,
                                             method=self.method, options=dict(step_size=self.step_size, perturb=self.perturb))
        else:
            out = torchdiffeq.odeint(self.odefunc, x, self.integration_time, rtol=self.rtol, atol=self.atol,
                                     method=self.method, options=dict(step_size=self.step_size, perturb=self.perturb))

        return out[-1]


class STBlock(nn.Module):

    def __init__(self, receptive_field, dilation, hidden_channels, dropout, method, time, step_size, alpha,
                 rtol, atol, adjoint, perturb=False):
        super(STBlock, self).__init__()
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
        self.gconv_1 = CGP(hidden_channels, hidden_channels, alpha=alpha,
                           method=method, time=time, step_size=step_size, rtol=rtol, atol=atol,
                           adjoint=adjoint, perturb=perturb)
        self.gconv_2 = CGP(hidden_channels, hidden_channels, alpha=alpha,
                           method=method, time=time, step_size=step_size, rtol=rtol, atol=atol,
                           adjoint=adjoint, perturb=perturb)

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

        x = F.dropout(x, self.dropout, training=self.training)

        x = self.gconv_1(x, self.graph) + self.gconv_2(x,
                                                       self.graph.transpose(1, 0))

        x = nn.functional.pad(x, (self.receptive_field - x.size(3), 0))

        return x

    def setGraph(self, graph):
        self.graph = graph

    def setIntermediate(self, dilation):
        self.new_dilation = dilation
        self.intermediate_seq_len = self.receptive_field


class MTGODE(nn.Module):

    def __init__(self, buildA_true, num_nodes, predefined_A=None, static_feat=None, dropout=0.3,
                 subgraph_size=20, node_dim=40, dilation_exponential=1, conv_channels=32, end_channels=128,
                 seq_length=12, in_dim=2, out_dim=12, tanhalpha=3, method_1='euler', time_1=1.2, step_size_1=0.4,
                 method_2='euler', time_2=1.0, step_size_2=0.25, alpha=1.0, rtol=1e-4, atol=1e-3, adjoint=False,
                 perturb=False, ln_affine=True):

        super(MTGODE, self).__init__()

        if method_1 == 'euler':
            self.integration_time = time_1
            self.estimated_nfe = round(self.integration_time / step_size_1)
        elif method_1 == 'rk4':
            self.integration_time = time_1
            self.estimated_nfe = round(
                self.integration_time / (step_size_1 / 4.0))
        else:
            raise ValueError("Oops! Temporal ODE solver is invaild.")

        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.seq_length = seq_length
        self.ln_affine = ln_affine
        self.adjoint = adjoint

        self.start_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=conv_channels, kernel_size=(1, 1))

        self.gc = graph_constructor(
            num_nodes, subgraph_size, node_dim, alpha=tanhalpha, static_feat=static_feat)
        self.idx = torch.arange(self.num_nodes)

        max_kernel_size = 7
        if dilation_exponential > 1:
            self.receptive_field = int(1 + (max_kernel_size - 1) * (
                dilation_exponential**self.estimated_nfe - 1) / (dilation_exponential - 1))
        else:
            self.receptive_field = self.estimated_nfe * \
                (max_kernel_size - 1) + 1

        if ln_affine:
            self.affine_weight = nn.Parameter(torch.Tensor(
                *(conv_channels, self.num_nodes)))  # C*H
            self.affine_bias = nn.Parameter(torch.Tensor(
                *(conv_channels, self.num_nodes)))  # C*H

        self.ODE = ODEBlock(ODEFunc(STBlock(receptive_field=self.receptive_field, dilation=dilation_exponential,
                                            hidden_channels=conv_channels, dropout=self.dropout, method=method_2,
                                            time=time_2, step_size=step_size_2, alpha=alpha, rtol=rtol, atol=atol,
                                            adjoint=False, perturb=perturb)),
                            method_1, step_size_1, rtol, atol, adjoint, perturb)

        self.end_conv_0 = nn.Conv2d(
            in_channels=conv_channels, out_channels=end_channels//2, kernel_size=(1, 1), bias=True)
        self.end_conv_1 = nn.Conv2d(
            in_channels=end_channels//2, out_channels=end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(
            in_channels=end_channels, out_channels=out_dim, kernel_size=(1, 1), bias=True)

        if ln_affine:
            self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.affine_weight)
        init.zeros_(self.affine_bias)

    def forward(self, history_data, future_data, batch_seen: int, epoch: int, train: bool, **kwargs):
        """Feedforward function of MSTE.

        Args:
            history_data: Tensor[B, L, N, C]
        Returns:
            output: Tensor[B, L, N, 1]
        """
        idx = None
        input = history_data.transpose(1, 3)

        seq_len = input.size(3)
        assert seq_len == self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length < self.receptive_field:
            input = nn.functional.pad(
                input, (self.receptive_field-self.seq_length, 0))

        if self.buildA_true:
            if idx is None:
                adp = self.gc(self.idx.to(input.device))
            else:
                adp = self.gc(idx)
        else:
            adp = self.predefined_A

        x = self.start_conv(input)

        if self.adjoint:
            self.ODE.odefunc.stnet.setIntermediate(dilation=1)
        self.ODE.odefunc.stnet.setGraph(adp)
        x = self.ODE(x, self.integration_time)
        self.ODE.odefunc.stnet.setIntermediate(dilation=1)

        x = x[..., -1:]
        x = F.layer_norm(x, tuple(x.shape[1:]),
                         weight=None, bias=None, eps=1e-5)

        if self.ln_affine:
            if idx is None:
                x = torch.add(torch.mul(x, self.affine_weight[:, self.idx].unsqueeze(
                    -1)), self.affine_bias[:, self.idx].unsqueeze(-1))  # C*H
            else:
                x = torch.add(torch.mul(
                    # C*H
                    x, self.affine_weight[:, idx].unsqueeze(-1)), self.affine_bias[:, idx].unsqueeze(-1))

        x = F.relu(self.end_conv_0(x))
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        return x
