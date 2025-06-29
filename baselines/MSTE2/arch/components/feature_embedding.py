from torch import nn
import torch


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
