import torch
from torch import nn
import torch.nn.functional as F

from .feature_embedding import FeatureEmbedding
from .time_ode import TimeODE, TimeODEFunc, TimeODENet


class ST_Encoder(nn.Module):
    def __init__(self,
                 num_nodes,
                 seq_length=12,
                 steps_per_day=288,
                 input_dim=3,
                 input_embedding_dim=24,
                 tod_embedding_dim=24,
                 dow_embedding_dim=24,
                 spatial_embedding_dim=0,
                 adaptive_embedding_dim=80,
                 time_1=1.0, step_size_1=0.25,
                 time_2=1.0, step_size_2=0.25,
                 feed_forward_dim=256,
                 num_heads=4,
                 dropout=0.1):
        super().__init__()
        method = 'euler'
        rtol = 1e-4
        atol = 1e-3
        perturb = False

        self.seq_length = seq_length

        self.enrichment = FeatureEmbedding(num_nodes,
                                           seq_length,
                                           input_dim,
                                           input_embedding_dim,
                                           tod_embedding_dim,
                                           dow_embedding_dim,
                                           steps_per_day,
                                           spatial_embedding_dim,
                                           adaptive_embedding_dim)
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )

        self.ODE = TimeODE(
            TimeODEFunc(
                TimeODENet(
                    self.model_dim,
                    feed_forward_dim,
                    num_heads,
                    dropout,
                    method,
                    time_2, step_size_2,
                    rtol, atol, perturb
                )
            ),
            method,
            time_1, step_size_1,
            rtol, atol, perturb
        )

    def forward(self, x):
        '''
        X: [B, T, N, C]
        '''
        x = self.enrichment(x)

        x = self.ODE(x)

        return x
