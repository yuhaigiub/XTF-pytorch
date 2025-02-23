import torch
import torch.optim as optim
from torch import nn
import util

def initiate_loss_fn(type):
    if type == "huber": return nn.HuberLoss()
    elif type == "mae": return util.masked_mae
    else: raise Exception("invalid loss fn")

class EngineXAI():
    def __init__(self,
                 scaler, 
                 model,
                 blackbox,
                 pertubate_node,
                 degreeCentrality,
                 closenessCentrality,
                 pertubate,
                 num_nodes, 
                 lrate, wdecay, 
                 device, 
                 adj_mx, 
                 train_loss="mae", test_loss="mae"):
        self.model = model
        self.model.to(device)
        self.blackbox = blackbox
        self.blackbox.to(device)
        self.pertubate_node = pertubate_node
        self.degreeCentrality = degreeCentrality
        self.closenessCentrality = closenessCentrality
        self.pertubate = pertubate
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.train_loss = initiate_loss_fn(train_loss)
        self.test_loss = initiate_loss_fn(test_loss)
        self.scaler = scaler
        self.clip = None
        self.edge_index = [[], []]
        self.edge_weight = []
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_mx.item((i, j)) != 0:
                    self.edge_index[0].append(i)
                    self.edge_index[1].append(j)
                    self.edge_weight.append(adj_mx.item((i, j)))

        self.adj_mx = torch.tensor(adj_mx).to(device)
        self.edge_index = torch.tensor(self.edge_index).to(device)
        self.edge_weight = torch.tensor(self.edge_weight).to(device)
        self.l2_coeff = 0.0001


    def train(self, X):
        '''
        X: [batch_size, channels, num_nodes, time_steps]
        
        NOTE: 
        - Y and Ym output must have shape: [batch_size, time_steps, num_nodes, out_dim]
        - Saliency output in engine: [batch_size, in_dim, num_nodes, time_steps] (after transpose)
        '''
        self.model.train()
        self.optimizer.zero_grad()
        Y = self.blackbox(X.transpose(1,3)) 
        Y = Y[:, :, :, 0:1]
        Ys = self.scaler.inverse_transform(Y)
        saliency = self.model(X.transpose(1,3))  
        saliency = saliency.transpose(-3, -1)
        top_k_spatial = 30
        top_k_temporal = 4
        X_node = self.degreeCentrality(saliency, self.adj_mx, top_k_spatial, True)
        Xm, _ = self.pertubate.apply(X, X_node, top_k_temporal, True)
        Ym = self.blackbox(Xm.transpose(1,3)) 
        Ym = Ym[:, :, :, 0:1]
        Yms = self.scaler.inverse_transform(Ym)
        
        l1_norm = sum(p.abs().sum() for p in self.model.parameters())
        l1_lambda = 0.0001
        loss = self.train_loss(Ym, Y) + l1_lambda * l1_norm
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        
        mape = util.masked_mape(Yms, Ys)
        rmse = util.masked_rmse(Ym, Y)
        return loss.item(), mape.item(), rmse.item()
    
    
    def eval(self, X):
        '''
        X: [batch_size, channels, num_nodes, time_steps]
        '''
        self.model.eval()
        Y = self.blackbox(X.transpose(1,3)) 
        Y = Y[:, :, :, 0:1]
        Ys = self.scaler.inverse_transform(Y)
        saliency = self.model(X.transpose(1,3)) 
        saliency = saliency.transpose(-3, -1)
        top_k_spatial = 30
        top_k_temporal = 4
        X_node = self.degreeCentrality(saliency, self.adj_mx, top_k_spatial, True)
        Xm, _ = self.pertubate.apply(X, X_node, top_k_temporal, True)
        Ym = self.blackbox(Xm.transpose(1,3))
        # [batch_size, time_steps, num_nodes, channels]
        Ym = Ym[:, :, :, 0:1]
        Yms = self.scaler.inverse_transform(Ym)
        l1_norm = sum(p.abs().sum() for p in self.model.parameters())
        l1_lambda = 0.0001
        loss = self.train_loss(Ym, Y) + l1_lambda * l1_norm
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(Yms, Ys)
        rmse = util.masked_rmse(Ym, Y)
        return loss.item(), mape.item(), rmse.item()
    
    
    def horizon_forward(self, X):
        self.model.eval()
        Y = self.blackbox(X.transpose(1,3)) 
        saliency = self.model(X.transpose(1,3)) 
        saliency = saliency.transpose(-3, -1)
        top_k_spatial = 30
        top_k_temporal = 4
        X_node = self.degreeCentrality(saliency, self.adj_mx, top_k_spatial, True)
        Xm, _ = self.pertubate.apply(X, X_node, top_k_temporal, True)
        Ym = self.blackbox(Xm.transpose(1,3))
        return Y, Ym