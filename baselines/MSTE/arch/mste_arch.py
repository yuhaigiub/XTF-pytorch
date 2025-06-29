import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

import torchdiffeq

from .components import ST_Encoder

'''
replace STEncoder = MTGODE
replace GCN = Spatial Self Attention
'''


class MSTE(nn.Module):
    def __init__(self,
                 n_experts: int, n_stacks: int,
                 time_0, step_size_0,
                 config,
                 decoder_types=[1, 1]):
        super().__init__()

        self.experts = nn.ModuleList()
        for _ in range(n_experts):
            expert = STExpert(n_stacks,
                              time_0, step_size_0,
                              config,
                              decoder_types)
            self.experts.append(expert)

    def forward(self, history_data: Tensor, future_data: Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> Tensor:
        """Feedforward function of MSTE.

        Args:
            history_data: Tensor[B, L, N, C]
        Returns:
            output: Tensor[B, L, N, 1]
        """
        X = history_data
        out = torch.zeros_like(X[..., 0:1]).type_as(X)
        for i in range(len(self.experts)):
            out = out + self.experts[i](X)
        return out


class STExpert(nn.Module):
    def __init__(self,
                 n_stacks,
                 time_0, step_size_0,
                 config,
                 decoder_types):
        super().__init__()
        self.stacks = nn.ModuleList()
        for _ in range(n_stacks):
            stack = STStack(time_0, step_size_0, config, decoder_types)
            self.stacks.append(stack)

        self.backcasts = []

    def forward(self, backcast: Tensor):
        '''
        backcast: [B, L, N, C]
        '''
        self.backcasts = []
        forecast = torch.zeros(
            size=[*(backcast.shape[:-1]), 1]).type_as(backcast)

        for stack in self.stacks:
            b, f = stack(backcast)
            self.backcasts.append(b)
            backcast = backcast - b
            forecast = forecast + f

        return forecast


class STStack(nn.Module):
    def __init__(self,
                 time_0, step_size_0,
                 config,
                 decoder_types):
        super().__init__()
        self.time_0 = time_0
        self.step_size_0 = step_size_0
        self.nfe = round(time_0 / step_size_0)

        # self.blockODE = STBlock(in_dim, seq_len,
        #                         conv_dim, end_dim,
        #                         time_1, step_size_1,
        #                         time_2, step_size_2)
        # self.blockODE.ODE.set_adj(adj_mx)

        self.blocks = nn.ModuleList()
        for i in range(self.nfe):
            block = STBlock(config, decoder_types)
            self.blocks.append(block)

    def forward(self, backcast: Tensor):
        '''
        backcast: [B, L, N, C]
        '''
        # self.blockODE.forecast = torch.zeros(size=[*(X.shape[:-1]), 1]).type_as(backcast)
        # time = torch.linspace(0, self.time_0, self.nfe + 1).float().to(backcast.device)
        # backcast = torchdiffeq.odeint(self.blockODE, backcast, time, method='euler')[-1]
        # return backcast, self.blockODE.forecast

        forecast = torch.zeros(
            size=[*(backcast.shape[:-1]), 1]).type_as(backcast)
        for block in self.blocks:
            b, f = block(None, backcast)
            backcast = backcast - b
            forecast = forecast + f
        return backcast, forecast


class STBlock(nn.Module):
    def __init__(self,
                 config,
                 decoder_types):
        super().__init__()
        self.st_encoder = ST_Encoder(**config)
        self.seq_len = config['seq_length']
        self.model_dim = (
            config['input_embedding_dim']
            + config['tod_embedding_dim']
            + config['dow_embedding_dim']
            + config['spatial_embedding_dim']
            + config['adaptive_embedding_dim']
        )

        self.backcast_decoder = nn.Sequential(
            nn.Conv2d(self.model_dim, self.model_dim // 2, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.model_dim // 2, self.seq_len, kernel_size=(1, 1))
        )
        self.forecast_decoder = nn.Sequential(
            nn.Conv2d(self.model_dim, self.model_dim // 2, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.model_dim // 2, self.seq_len, kernel_size=(1, 1))
        )

    def forward(self, t, x: Tensor):
        '''
        x: [b, t, n, c]
        '''
        x = self.st_encoder(x)

        backcast = self.backcast_decoder(x)
        _z = torch.zeros_like(backcast)
        backcast = torch.cat([backcast, _z, _z], dim=-1)

        forecast = self.forecast_decoder(x)

        return backcast, forecast
