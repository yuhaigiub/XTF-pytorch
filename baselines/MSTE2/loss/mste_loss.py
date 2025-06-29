import numpy as np
import torch
import torch.nn.functional as F
from basicts.metrics import masked_mae, masked_mse
import torch.distributed as dist


class mste_loss():
    def __init__(self, w2, w3, use_huber=True, delta=1.0):
        self.w2 = w2
        self.w3 = w3
        self.use_huber = use_huber
        self.delta = delta

    def __call__(self,
                 prediction: torch.Tensor, target: torch.Tensor,
                 inputs: torch.Tensor = None, model=None,
                 null_val: float = np.nan):
        if self.use_huber:
            loss = self.masked_huber(
                prediction, target, null_val, delta=self.delta)
        else:
            loss = masked_mae(prediction, target, null_val)

        if model is None:
            l2 = 0
            l3 = 0
        else:
            try:
                experts = model.module.experts
            except:
                experts = model.experts

            l2 = torch.tensor(0.0, device=prediction.device)
            for expert in experts:
                for b in expert.backcasts:
                    # choose a & t
                    a = np.random.randint(0, 4)
                    t = np.random.choice([1, 2, 3, 4, 5])
                    l2 = l2 + \
                        masked_mse(
                            inputs[:, a::t, ...], b[:, a::t, ...], null_val)
            l2 = self.sync_loss(l2)

            l3 = torch.tensor(0.0, device=prediction.device)
            n = len(experts)
            for i in range(0, n-1):
                for j in range(i+1, n):
                    model1 = experts[i]
                    model2 = experts[j]
                    l2_norm = torch.tensor(0.0)

                    params_A = model1.state_dict()
                    params_B = model2.state_dict()

                    all_params_A = torch.cat([
                        param.flatten() for param in params_A.values()])
                    all_params_B = torch.cat([
                        param.flatten() for param in params_B.values()])

                    l2_norm = l2_norm + torch.norm(all_params_A - all_params_B)

                    # Take the square root of the accumulated squared norms
                    l3 = l3 + torch.sqrt(l2_norm)
            l3 = self.sync_loss(l3)

        loss = loss + self.w2 * l2 + self.w3 * l3
        return loss

    def sync_loss(self, loss: torch.Tensor):
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= dist.get_world_size()

        return loss

    def masked_huber(self, prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan, delta=1.0) -> torch.Tensor:
        if np.isnan(null_val):
            mask = ~torch.isnan(target)
        else:
            eps = 5e-5
            mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(
                target).to(target.device), atol=eps, rtol=0.0)

        mask = mask.float()
        # Normalize mask to avoid bias in the loss due to the number of valid entries
        mask /= torch.mean(mask)
        mask = torch.nan_to_num(mask)  # Replace any NaNs in the mask with zero

        loss = F.huber_loss(prediction, target, reduction='none', delta=delta)
        loss = loss * mask  # Apply the mask to the loss
        loss = torch.nan_to_num(loss)  # Replace any NaNs in the loss with zero

        return torch.mean(loss)
