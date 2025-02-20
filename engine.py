import numpy as np
import torch
from torch import optim
import util

class Engine():
    def __init__(self, 
                 scaler, 
                 model,
                 lrate, wdecay, w2, w3,
                 device,
                 grad_acc_step_max,
                 clip,
                 huber_delta=1.0,):
        self.model = model
        self.model.to(device)
        
        self.lrate = lrate
        self.wdecay = wdecay
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        print('number of parameters:', sum(p.numel() for p in model.parameters()))
        
        self.loss = lambda o, l: util.masked_huber(o, l, huber_delta)
        self.scaler = scaler
        self.clip = clip
        self.w2 = w2
        self.w3 = w3
        
        self.grad_acc_step = 0
        self.grad_acc_step_max = grad_acc_step_max
    
    def train(self, input, real):
        '''
        input: [B, L, N, C]
        '''
        self.model.train()
        self.grad_acc_step += 1
        
        output = self.model(input)
        predict = self.scaler.inverse_transform(output)
        
        l1 = self.loss(predict, real)
        
        l2 = torch.tensor(0.0)
        for expert in self.model.experts:
            for b in expert.backcasts:
                # choose a & t
                a = np.random.randint(0, 4)
                t = np.random.choice([1, 2, 3, 4, 5])
                l2 = l2 + util.masked_mae(input[:, a::t,...], b[:, a::t,...])
        
        l3 = torch.tensor(0.0)
        n = len(self.model.experts)
        for i in range(0, n-1):
            for j in range(i+1, n):
                model1 = self.model.experts[i]
                model2 = self.model.experts[j]
                l2_norm = torch.tensor(0.0)
                
                params_A = model1.state_dict()
                params_B = model2.state_dict()
                
                all_params_A = torch.cat([param.flatten() for param in params_A.values()])
                all_params_B = torch.cat([param.flatten() for param in params_B.values()])
                
                l2_norm = l2_norm + torch.norm(all_params_A - all_params_B)
                
                # # Loop through the parameters of both models and compute the L2 norm
                # for param1, param2 in zip(model1.parameters(), model2.parameters()):
                #     diff = param1 - param2  # Compute difference between weights
                #     l2_norm = l2_norm + torch.norm(diff) ** 2  # Accumulate squared norms
                
                # Take the square root of the accumulated squared norms
                l3 = l3 + torch.sqrt(l2_norm)
        
        loss = l1 + self.w2 * l2 + self.w3 * l3
        mae = loss.item()
        
        # normalize loss (for gradient accumulation)
        loss = loss / self.grad_acc_step_max
        
        loss.backward()
        
        if self.grad_acc_step == self.grad_acc_step_max:
            if self.clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            self.grad_acc_step = 0
        
        mape = util.masked_mape(predict, real).item()
        rmse = util.masked_rmse(predict, real).item()
        return mae, mape, rmse, l2.item() * self.w2, l3.item() * self.w3
    
    def eval(self, input, real):
        '''
        input: [B, L, N, C]
        '''
        self.model.eval()
        with torch.no_grad():
            output = self.model(input)
            predict = self.scaler.inverse_transform(output)
            
            metrics = util.metric(predict, real)
            return metrics