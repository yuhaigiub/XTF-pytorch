import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .components import ST_Encoder


class MSTE(nn.Module):
    def __init__(self,
                 n_experts: int, n_stacks: int,
                 time_0, step_size_0,
                 config):
        super().__init__()

        self.experts = nn.ModuleList()
        for _ in range(n_experts):
            expert = STExpert(n_stacks, time_0, step_size_0, config)
            self.experts.append(expert)

    def forward(self,
                history_data: Tensor,
                future_data: Tensor = None,
                batch_seen: int = 0,
                epoch: int = 0,
                train: bool = True,
                **kwargs: dict
                ) -> Tensor:
        '''history_data: Tensor[B, T, N, C]'''

        X = history_data
        out = torch.zeros_like(X[..., 0:1]).type_as(X)
        for i in range(len(self.experts)):
            out = out + self.experts[i](X)

        return out  # [B, T, N, 1]


class STExpert(nn.Module):
    def __init__(self, n_stacks, time_0, step_size_0, config):
        super().__init__()
        self.stacks = nn.ModuleList()
        for _ in range(n_stacks):
            stack = STStack(time_0, step_size_0, config)
            self.stacks.append(stack)

        self.backcasts = []

    def forward(self, backcast: Tensor) -> Tensor:
        '''backcast: [B, T, N, C]'''

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
    def __init__(self, time_0, step_size_0, config):
        super().__init__()
        self.time_0 = time_0
        self.step_size_0 = step_size_0
        self.nfe = round(time_0 / step_size_0)

        self.blocks = nn.ModuleList()
        for _ in range(self.nfe):
            block = STBlock(config)
            self.blocks.append(block)

    def forward(self, backcast: Tensor) -> Tensor:
        '''backcast: [B, T, N, C]'''

        forecast = torch.zeros(size=[*(backcast.shape[:-1]), 1]).type_as(backcast)
        for block in self.blocks:
            b, f = block(None, backcast)
            backcast = backcast - b
            forecast = forecast + f
        return backcast, forecast


class STBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.st_encoder = ST_Encoder(**config)
        self.input_dim = config['input_dim']
        self.seq_len = config['seq_length']
        self.model_dim = (
            config['input_embedding_dim']
            + config['tod_embedding_dim']
            + config['dow_embedding_dim']
            + config['spatial_embedding_dim']
            + config['adaptive_embedding_dim']
        )

        self.backcast_decoder = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.model_dim // 2, self.input_dim)
        )
        self.forecast_decoder = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.model_dim // 2, 1)
        )

    def forward(self, t, x: Tensor) -> Tensor:
        '''x: [B, T, N, C]'''

        x = self.st_encoder(x)
        backcast = self.backcast_decoder(x)
        # _z = torch.zeros_like(backcast)

        # backcast = torch.cat([backcast, _z, _z], dim=-1)
        forecast = self.forecast_decoder(x)

        return backcast, forecast
