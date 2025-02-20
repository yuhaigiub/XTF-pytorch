import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

import torchdiffeq

class MSTE(nn.Module):
    def __init__(self, 
                 n_experts: int, 
                 n_stacks: int, 
                 in_dim, seq_len,
                 conv_dim, end_dim,
                 adj_mx,
                 time_0, step_size_0,
                 time_1, step_size_1,
                 time_2, step_size_2,
                 decoder_types=[1, 1],
                 dropout=0.1):
        super().__init__()
        
        self.weights = nn.Parameter(torch.ones(n_experts, requires_grad=True))
        
        self.experts = nn.ModuleList()
        # self.linears = nn.ModuleList()
        for _ in range(n_experts):
            expert = STExpert(n_stacks, 
                              in_dim, seq_len,
                              conv_dim, end_dim,
                              adj_mx,
                              time_0, step_size_0,
                              time_1, step_size_1,
                              time_2, step_size_2,
                              decoder_types, 
                              dropout)
            self.experts.append(expert)
    
    def forward(self, X: Tensor) -> Tensor:
        out = []
        
        for i in range(len(self.experts)):
            y: Tensor = self.experts[i](X)
            # out = out + self.linears[i](y)
            out.append(y)
        
        weights = self.weights.view(-1, 1, 1, 1, 1)
        out = torch.stack(out, dim=0)
        
        out = torch.sum(weights * out, dim=0)
        
        return out

class STExpert(nn.Module):
    def __init__(self, n_stacks, 
                 in_dim, seq_len,
                 conv_dim, end_dim,
                 adj_mx,
                 time_0, step_size_0,
                 time_1, step_size_1,
                 time_2, step_size_2,
                 decoder_types,
                 dropout):
        super().__init__()
        self.stacks = nn.ModuleList()
        for _ in range(n_stacks):
            stack = STStack(in_dim, seq_len,
                            conv_dim, end_dim,
                            adj_mx,
                            time_0, step_size_0,
                            time_1, step_size_1,
                            time_2, step_size_2,
                            decoder_types,
                            dropout)
            self.stacks.append(stack)
        
        self.backcasts = []
    
    def forward(self, backcast: Tensor):
        '''
        backcast: [B, L, N, C]
        '''
        self.backcasts = []
        forecast = torch.zeros(size=[*(backcast.shape[:-1]), 1]).type_as(backcast)
        
        for stack in self.stacks:
            b, f = stack(backcast)
            self.backcasts.append(b)
            backcast = backcast - b
            forecast = forecast + f
        
        return forecast

class STStack(nn.Module):
    def __init__(self, 
                 in_dim: int, seq_len: int, 
                 conv_dim: int, end_dim: int,
                 adj_mx: Tensor,
                 time_0, step_size_0,
                 time_1, step_size_1,
                 time_2, step_size_2,
                 decoder_types,
                 dropout):
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
            block = STBlock(in_dim, seq_len,
                            conv_dim, end_dim,
                            time_1, step_size_1, 
                            time_2, step_size_2,
                            decoder_types,
                            dropout)
            block.ODE.set_adj(adj_mx)
            self.blocks.append(block)
    
    def forward(self, backcast: Tensor):
        '''
        backcast: [B, L, N, C]
        '''
        # self.blockODE.forecast = torch.zeros(size=[*(X.shape[:-1]), 1]).type_as(backcast)
        # time = torch.linspace(0, self.time_0, self.nfe + 1).float().to(backcast.device)
        # backcast = torchdiffeq.odeint(self.blockODE, backcast, time, method='euler')[-1]
        # return backcast, self.blockODE.forecast
        
        forecast = torch.zeros(size=[*(backcast.shape[:-1]), 1]).type_as(backcast)
        for block in self.blocks:
            b, f = block(None, backcast)
            backcast = backcast - b
            forecast = forecast + f
        return backcast, forecast

class STBlock(nn.Module):
    def __init__(self,
                 in_dim, seq_len,
                 conv_dim, end_dim,
                 time_1, step_size_1,
                 time_2, step_size_2,
                 decoder_types,
                 dropout):
        super().__init__()
        self.in_dim = in_dim
        self.seq_len = seq_len
        self.conv_dim = conv_dim
        self.end_dim = end_dim
        
        self.time_1 = time_1
        self.step_size_1 = step_size_1
        
        self.nfe = round(time_1 / step_size_1)
        max_kernel_size = 7
        self.receptive_field = self.nfe * (max_kernel_size - 1) + in_dim
        self.receptive_field = max(self.seq_len, self.receptive_field)
        
        self.start_conv = nn.Conv2d(in_dim, conv_dim, kernel_size=(1, 1))
        
        self.ODE = STEncoder(self.receptive_field, 1, conv_dim, time_2, step_size_2, dropout)
        
        self.backcast_decoder = self._create_decoder(decoder_types[0])
        self.forecast_decoder = self._create_decoder(decoder_types[1])
        
        self.forecast = None
    
    def _create_decoder(self, decoder_type: int):
        if decoder_type == 1:
            return nn.Sequential(
                nn.Conv2d(self.conv_dim, self.end_dim, kernel_size=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(self.end_dim, self.seq_len, kernel_size=(1, 1)),
            )
        elif decoder_type == 2:
            return nn.Sequential(
                nn.Conv2d(self.conv_dim, self.end_dim *2, kernel_size=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(self.end_dim * 2, self.end_dim, kernel_size=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(self.end_dim, self.seq_len, kernel_size=(1, 1)),
            )
        else:
            raise Exception("Invalid decoder type")
    
    def forward(self, t, X: Tensor):
        '''
            X: [B, L, N, C]
        '''
        
        X = X.transpose(1, 3) # [B, C, N, L]
        if self.seq_len < self.receptive_field:
            X = F.pad(X, (self.receptive_field - self.seq_len, 0))
        X = self.start_conv(X)
        
        time = torch.linspace(self.time_1, 0.0, self.nfe+1).float().to(X.device)
        X = torchdiffeq.odeint(self.ODE, X, time, method='euler')[-1]
        self.ODE.set_intermediate(1)
        
        X = X[..., -self.in_dim:]
        
        X = F.layer_norm(X, tuple(X.shape[1:]), weight=None, bias=None, eps=1e-5)
        
        backcast = self.backcast_decoder(X) # [B, L, N, C]
        forecast = self.forecast_decoder(X)[..., -1:] # [B, L, N, 1]
        
        # self.forecasts.append(forecast)
        # self.forecast = self.forecast + forecast
        # return backcast
        
        return backcast, forecast

class STEncoder(nn.Module):
    def __init__(self,
                 receptive_field, dilation,
                 hidden_dim,
                 time_2, step_size_2,
                 dropout):
        super().__init__()
        self.dropout= dropout
        self.receptive_field = receptive_field
        self.intermediate_seq_len = receptive_field
        self.new_dilation = 1
        self.dilation_factor = dilation
        
        self.inception_1 = DilatedInception(hidden_dim, hidden_dim)
        self.inception_2 = DilatedInception(hidden_dim, hidden_dim)
        
        self.gconv_1 = CGPODE(hidden_dim, hidden_dim, time_2, step_size_2)
        self.gconv_2 = CGPODE(hidden_dim, hidden_dim, time_2, step_size_2)
        
        self.adj = None
    
    def forward(self, t, x: Tensor):
        x = x[..., -self.intermediate_seq_len:]
        for tconv in self.inception_1.tconv:
            tconv.dilation = (1, self.new_dilation)
        for tconv in self.inception_2.tconv:
            tconv.dilation = (1, self.new_dilation)
        
        _filter = self.inception_1(x)
        _filter = torch.tanh(_filter)
        
        _gate = self.inception_2(x)
        _gate = torch.sigmoid(_gate)
        
        x = _filter * _gate
        
        self.new_dilation *= self.dilation_factor
        self.intermediate_seq_len = x.size(3)
        
        x = F.dropout(x, self.dropout)
        x = self.gconv_1(x, self.adj) + self.gconv_2(x, self.adj.T)
        
        x = F.pad(x, (self.receptive_field - x.size(3), 0))
        return x
    
    def set_adj(self, adj: Tensor):
        adj = adj + torch.eye(adj.size(0)).to(adj.device)
        d = adj.sum(1)
        _d = torch.diag(torch.pow(d, -0.5))
        
        self.adj = torch.mm(torch.mm(_d, adj), _d)
    
    def set_intermediate(self, dilation):
        self.new_dilation = dilation
        self.intermediate_seq_len = self.receptive_field

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x: Tensor, A: Tensor):
        # x.shape = (batch, channels, nodes, time_steps)
        # A.shape = (node, node)
        x = torch.einsum('ncwl,vw->ncvl', (x, A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super(linear,self).__init__()
        self.mlp = nn.Conv2d(in_dim, 
                             out_dim, 
                             kernel_size=(1, 1), 
                             padding=(0, 0), 
                             stride=(1, 1), 
                             bias=bias)
    
    def forward(self, x: Tensor):
        return self.mlp(x)

class DilatedInception(nn.Module):
    def __init__(self, in_dim, out_dim, dilation=1):
        super(DilatedInception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernels = [2, 3, 5, 7]
        self.n = len(self.kernels)
        
        assert out_dim % self.n == 0, 'out_dim % kernels != 0'
        
        out_dim = int(out_dim / self.n)
        
        for kernel in self.kernels:
            self.tconv.append(nn.Conv2d(in_dim, out_dim, (1, kernel), dilation=(1, dilation)))
    
    def forward(self, input: Tensor):
        x = []
        for i in range(self.n):
            output = self.tconv[i](input)
            x.append(output)
        
        for i in range(self.n):
            x[i] = x[i][..., -x[-1].size(3):]
        
        x = torch.cat(x, dim=1)
        return x

class CGPODE(nn.Module):
    def __init__(self, in_dim, out_dim, time, step_size, alpha=2.0):
        super(CGPODE, self).__init__()
        self.time = time
        self.step_size = step_size
        
        self.nfe = round(time / step_size)
        
        self.odefunc = CGPFunc(alpha)
        self.mlp = linear((self.nfe + 1) * in_dim, out_dim)
    
    def forward(self, x: Tensor, adj: Tensor):
        # self.odefunc.set_x0(x)
        self.odefunc.set_adj(adj)
        
        t = torch.linspace(0, self.time, self.nfe+1).float().type_as(x)
        out = torchdiffeq.odeint(self.odefunc, x, t, method="euler")
        
        outs = self.odefunc.out
        self.odefunc.out = []
        outs.append(out[-1])
        h_out = torch.cat(outs, dim=1)
        h_out = self.mlp(h_out)
        return h_out
        
        # return out[-1]

class CGPFunc(nn.Module):
    def __init__(self, alpha=2.0):
        super(CGPFunc, self).__init__()
        self.out = []
        self.x0 = None
        self.adj = None
        self.alpha = alpha
        
        self.conv = nconv()
    
    def set_x0(self, x0: Tensor):
        self.x0 = x0.clone().detach()
    
    def set_adj(self, adj: Tensor):
        self.adj = adj
    
    def forward(self, t, x: Tensor):
        self.out.append(x)
        
        ax = self.conv(x, self.adj)
        
        return ax
