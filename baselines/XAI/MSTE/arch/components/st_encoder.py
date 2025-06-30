import torch
import torch.nn.functional as F
from torch import nn
from .st_block import STODE, ODEBlock, ODEFunc


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
                 dilation_exponential=1,
                 feed_forward_dim=256,
                 num_heads=4,
                 dropout=0.1):
        super().__init__()
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

        # default method1 = euler
        self.integration_time = time_1
        self.estimated_nfe = round(self.integration_time / step_size_1)

        max_kernel_size = 7
        if dilation_exponential > 1:
            self.receptive_field = int(1 + (max_kernel_size - 1) * (
                dilation_exponential**self.estimated_nfe - 1) / (dilation_exponential - 1))
        else:
            self.receptive_field = self.estimated_nfe * \
                (max_kernel_size - 1) + 1

        self.ODE = ODEBlock(
            ODEFunc(
                STODE(receptive_field=self.receptive_field,
                      dilation=dilation_exponential,
                      hidden_channels=self.model_dim,
                      feed_forward_dim=feed_forward_dim,
                      num_heads=num_heads,
                      dropout=dropout,
                      method='euler',
                      time=time_2, step_size=step_size_2,
                      rtol=1e-4, atol=1e-3, adjoint=False, perturb=False)),
            'euler', step_size_1, rtol=1e-4, atol=1e-3, adjoint=False, perturb=False)

    def forward(self, x):
        '''
        x: [b, t, n, c]
        '''
        x = self.enrichment(x)

        # (b, c, n, t)
        x = x.transpose(1, 3)
        seq_len = x.size(3)
        assert seq_len == self.seq_length, f'input sequence length not equal to preset sequence length'

        if self.seq_length < self.receptive_field:
            x = nn.functional.pad(x, (self.receptive_field-self.seq_length, 0))

        x = self.ODE(x, self.integration_time)
        self.ODE.odefunc.stnet.setIntermediate(dilation=1)

        x = x[..., -1:]
        x = F.layer_norm(
            x, tuple(x.shape[1:]), weight=None, bias=None, eps=1e-5)

        return x


class FeatureEmbedding(nn.Module):
    def __init__(self,
                 num_nodes,
                 in_steps,
                 input_dim,
                 input_embedding_dim,
                 tod_embedding_dim,
                 dow_embedding_dim,
                 steps_per_day,
                 spatial_embedding_dim,
                 adaptive_embedding_dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)

        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(
                    in_steps, num_nodes, adaptive_embedding_dim))
            )

    def forward(self, x):
        batch_size = x.shape[0]

        if self.tod_embedding_dim > 0:
            tod = x[..., 1] * self.steps_per_day
        if self.dow_embedding_dim > 0:
            dow = x[..., 2] * 7

        x = x[..., : self.input_dim]
        x = self.input_proj(x)

        # build features
        features = [x]
        if self.tod_embedding_dim > 0:
            # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            tod_emb = self.tod_embedding(tod.long())
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            dow_emb = self.dow_embedding(dow.long())
            features.append(dow_emb)
        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(
                batch_size, self.in_steps, *self.node_emb.shape)
            features.append(spatial_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)

        # (batch_size, in_steps, num_nodes, model_dim)
        x = torch.cat(features, dim=-1)

        return x
