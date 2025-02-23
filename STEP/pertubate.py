import sys
import os

import torch
import networkx as nx
import torch.nn.functional as F
import torch.nn as nn
from captum.attr import DeepLift, GradientShap, Lime, DeepLiftShap
import shap

from abc import ABC, abstractmethod

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
import util


class Pertubation(ABC):
    @abstractmethod
    def __init__(self, device, eps=1.0e-7):
        self.mask_tensor = None
        self.device = device
        self.eps = eps
    
    @abstractmethod
    def apply(self, X, mask_tensor):
        if X is None or mask_tensor is None:
            raise NameError("The mask_tensor should be fitted before or while calling the apply() method")


class FadeMovingAverage(Pertubation):
    def __init__(self, device, eps=1.0e-7,alpha_init=0.9):
        super().__init__(device, eps)
        self.alpha = torch.tensor(alpha_init, requires_grad=True, device=device)
    
    def apply(self, X, mask_tensor):
        '''
        X: [batch_size, channels, num_nodes, time_steps]
        mask_tensor: [batch_size, channels, num_nodes, time_steps]
        '''
        super().apply(X, mask_tensor)
        # average time_steps
        moving_average = torch.mean(X, dim=-1).to(self.device)
        moving_average = torch.unsqueeze(moving_average, -1)
        moving_average_tilted = moving_average.repeat(1, 1, 1, X.size(-1))
        X_pert = mask_tensor * X + (1 - mask_tensor) * moving_average_tilted
        return X_pert
    
    def optimize_step(self, optimizer, clip=None):
        optimizer.zero_grad()
        self.alpha.backward()
        optimizer.step()
        if clip is not None:
            if isinstance(clip, (tuple, list)):
                self.alpha.data.clamp_(min=clip[0], max=clip[1])


class GaussianBlur():
    """This class allows to create and apply 'Gaussian blur' perturbations on inputs based on masks.

    Attributes:
        mask_tensor (torch.tensor): The mask tensor than indicates the intensity of the perturbation
            to be applied at each input entry.
        eps (float): Small number used for numerical stability.
        device: Device on which the tensor operations are executed.
        sigma_max (float): Maximal width for the Gaussian blur.
    """
    def __init__(self, device, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std
        self.device = device
        
    def apply(self, X, saliency):
        tensor = torch.empty(*X.shape, device=self.device)
        tensor.normal_(self.mean, self.std)
        # here we consider important nodes to have saliency score = 0 (no pertubation)
        X_perturbed = saliency * tensor + (1 - saliency) * X
        # X_perturbed.to(self.device)
        return X_perturbed


def perturbate_node(M, adjacency_matrix, top_k, status):
    degree = torch.sum(adjacency_matrix, dim=1)
    degree = degree.view(1, 1, -1, 1)
    perturbed_M = M * degree
    flattened_M = perturbed_M.view(perturbed_M.size(0), perturbed_M.size(1), -1, perturbed_M.size(-1))
    _, indices = torch.topk(flattened_M, top_k, dim=2, largest=status, sorted=False)
    mask = torch.zeros_like(flattened_M, dtype=torch.bool)
    mask.scatter_(2, indices, True)
    top_k_M = flattened_M * mask
    top_k_M = top_k_M.view(perturbed_M.size(0), perturbed_M.size(1), -1, perturbed_M.size(-1))
    epsilon = 1e-6
    top_k_M = torch.where(torch.abs(top_k_M) < epsilon, torch.tensor(0.0), top_k_M)
    return top_k_M


def degreeCentrality(M, adjacency_matrix, top_k, status):
    adj_numpy = adjacency_matrix.cpu().numpy()
    G = nx.from_numpy_array(adj_numpy)
    degrees = dict(G.degree())
    degree_values = torch.tensor(list(degrees.values()), dtype=M.dtype, device=M.device)
    degree_values = degree_values.view(1, 1, -1, 1)
    perturbed_M = M * degree_values
    flattened_M = perturbed_M.view(perturbed_M.size(0), perturbed_M.size(1), -1, perturbed_M.size(-1))
    _, indices = torch.topk(flattened_M, top_k, dim=2, largest=status, sorted=False)
    mask = torch.zeros_like(flattened_M, dtype=torch.bool)
    mask.scatter_(2, indices, True)
    top_k_M = flattened_M * mask
    top_k_M = top_k_M.view(perturbed_M.size(0), perturbed_M.size(1), -1, perturbed_M.size(-1))
    epsilon = 1e-6
    top_k_M = torch.where(torch.abs(top_k_M) < epsilon, torch.tensor(0.0, device=M.device), top_k_M)
    return top_k_M


def perturbate_node_random(M, top_k):
    flattened_M = M.view(M.size(0), M.size(1), -1, M.size(-1))
    num_elements = flattened_M.size(2)
    random_indices = torch.randperm(num_elements, device=flattened_M.device)[:top_k]
    random_indices = random_indices.view(1, 1, top_k, 1).expand(flattened_M.size(0), flattened_M.size(1), top_k, flattened_M.size(-1))
    mask = torch.zeros_like(flattened_M, dtype=torch.bool, device=flattened_M.device)
    mask.scatter_(2, random_indices, True)
    top_k_M = flattened_M * mask
    top_k_M = top_k_M.view(M.size(0), M.size(1), -1, M.size(-1))
    epsilon = 1e-6
    top_k_M = torch.where(torch.abs(top_k_M) < epsilon, torch.tensor(0.0, device=top_k_M.device), top_k_M)
    return top_k_M


def closenessCentrality(M, adjacency_matrix, top_k, status):
    adj_numpy = adjacency_matrix.cpu().numpy()
    G = nx.from_numpy_array(adj_numpy)
    closeness_centrality_dict = nx.closeness_centrality(G)
    closeness_centrality = torch.tensor(list(closeness_centrality_dict.values()), dtype=M.dtype, device=M.device)
    closeness_centrality = closeness_centrality.view(1, 1, -1, 1)
    perturbed_M = M * closeness_centrality
    flattened_M = perturbed_M.view(perturbed_M.size(0), perturbed_M.size(1), -1, perturbed_M.size(-1))
    _, indices = torch.topk(flattened_M, top_k, dim=2, largest=status, sorted=False)
    mask = torch.zeros_like(flattened_M, dtype=torch.bool)
    mask.scatter_(2, indices, True)
    top_k_M = flattened_M * mask
    top_k_M = top_k_M.view(perturbed_M.size(0), perturbed_M.size(1), -1, perturbed_M.size(-1))
    epsilon = 1e-6
    top_k_M = torch.where(torch.abs(top_k_M) < epsilon, torch.tensor(0.0), top_k_M)
    return top_k_M


def betweennessCentrality(M, adjacency_matrix, top_k, status):
    adj_numpy = adjacency_matrix.cpu().numpy()
    G = nx.from_numpy_array(adj_numpy)
    betweenness_centrality_dict = nx.betweenness_centrality(G)
    betweenness_centrality = torch.tensor(list(betweenness_centrality_dict.values()), dtype=M.dtype, device=M.device)
    betweenness_centrality = betweenness_centrality.view(1, 1, -1, 1)
    perturbed_M = M * betweenness_centrality
    flattened_M = perturbed_M.view(perturbed_M.size(0), perturbed_M.size(1), -1, perturbed_M.size(-1))
    _, indices = torch.topk(flattened_M, top_k, dim=2, largest=status, sorted=False)
    mask = torch.zeros_like(flattened_M, dtype=torch.bool)
    mask.scatter_(2, indices, True)
    top_k_M = flattened_M * mask
    top_k_M = top_k_M.view(perturbed_M.size(0), perturbed_M.size(1), -1, perturbed_M.size(-1))
    epsilon = 1e-6
    top_k_M = torch.where(torch.abs(top_k_M) < epsilon, torch.tensor(0.0), top_k_M)
    return top_k_M


def eigenvalueCentrality(M, adjacency_matrix, top_k, status):
    adj_numpy = adjacency_matrix.cpu().numpy()
    G = nx.from_numpy_array(adj_numpy)
    eigenvector_centrality_dict = nx.eigenvector_centrality_numpy(G)
    eigenvector_centrality = torch.tensor(list(eigenvector_centrality_dict.values()), dtype=M.dtype, device=M.device)
    eigenvector_centrality = eigenvector_centrality.view(1, 1, -1, 1)
    perturbed_M = M * eigenvector_centrality
    flattened_M = perturbed_M.view(perturbed_M.size(0), perturbed_M.size(1), -1, perturbed_M.size(-1))
    _, indices = torch.topk(flattened_M, top_k, dim=2, largest=status, sorted=False)
    mask = torch.zeros_like(flattened_M, dtype=torch.bool)
    mask.scatter_(2, indices, True)
    top_k_M = flattened_M * mask
    top_k_M = top_k_M.view(perturbed_M.size(0), perturbed_M.size(1), -1, perturbed_M.size(-1))
    epsilon = 1e-6
    top_k_M = torch.where(torch.abs(top_k_M) < epsilon, torch.tensor(0.0), top_k_M)
    return top_k_M

def pageRank(M, adjacency_matrix, top_k, status):
    adj_numpy = adjacency_matrix.cpu().numpy()
    G = nx.from_numpy_array(adj_numpy)
    pagerank_dict = nx.pagerank(G)
    pagerank = torch.tensor(list(pagerank_dict.values()), dtype=M.dtype, device=M.device)
    pagerank = pagerank.view(1, 1, -1, 1)
    perturbed_M = M * pagerank
    flattened_M = perturbed_M.view(perturbed_M.size(0), perturbed_M.size(1), -1, perturbed_M.size(-1))
    _, indices = torch.topk(flattened_M, top_k, dim=2, largest=status, sorted=False)
    mask = torch.zeros_like(flattened_M, dtype=torch.bool)
    mask.scatter_(2, indices, True)
    top_k_M = flattened_M * mask
    top_k_M = top_k_M.view(perturbed_M.size(0), perturbed_M.size(1), -1, perturbed_M.size(-1))
    epsilon = 1e-6
    top_k_M = torch.where(torch.abs(top_k_M) < epsilon, torch.tensor(0.0), top_k_M)
    return top_k_M


class FadeMovingAverage_TopK(Pertubation):
    def __init__(self, device, eps=1.0e-7, alpha_init=0.9):
        super().__init__(device, eps)
        self.alpha = torch.tensor(alpha_init, requires_grad=True, device=device)
    
    def adjust_mask_tensor(self, mask_tensor, num_nodes):
        '''
        Adjusts mask_tensor to have the correct number of nodes.
        '''
        current_nodes = mask_tensor.size(2)
        if current_nodes < num_nodes:
            pad_size = num_nodes - current_nodes
            mask_tensor = torch.cat([mask_tensor, torch.zeros(mask_tensor.size(0), mask_tensor.size(1), pad_size, mask_tensor.size(3), device=mask_tensor.device)], dim=2)
        elif current_nodes > num_nodes:
            mask_tensor = mask_tensor[:, :, :num_nodes, :]
        return mask_tensor

    def apply(self, X, mask_tensor, top_k, status):
        '''
        X: [batch_size, channels, num_nodes, time_steps]
        mask_tensor: [batch_size, channels, num_nodes, time_steps]
        top_k: number of top time steps to select based on influence
        '''
        super().apply(X, mask_tensor)
        # Ensure mask_tensor has the correct shape along the num_nodes dimension
        mask_tensor = self.adjust_mask_tensor(mask_tensor, X.size(2))
        # Calculate the moving average across time steps
        moving_average = torch.mean(X, dim=-1, keepdim=True)  # [batch_size, channels, num_nodes, 1]
        moving_average_tilted = moving_average.expand_as(X)  # [batch_size, channels, num_nodes, time_steps]
        # Expand mask_tensor to match X's shape
        mask_tensor_expanded = mask_tensor.expand_as(X)
        # Apply perturbation
        X_pert = mask_tensor_expanded * X + (1 - mask_tensor_expanded) * moving_average_tilted
        # Calculate the influence scores for each time step
        influence_scores = torch.abs(X - moving_average_tilted)  # [batch_size, channels, num_nodes, time_steps]
        influence_scores = torch.mean(influence_scores, dim=(0, 1, 2))  # Average across batch_size, channels, and nodes
        _, top_indices = torch.topk(influence_scores, k=top_k, largest=status)
        top_k_mask = torch.zeros_like(X, dtype=torch.bool)
        top_k_mask[:, :, :, top_indices] = True
        X_pert_top_k = torch.where(top_k_mask, X_pert, torch.zeros_like(X))
        return X_pert_top_k, top_indices
    
    
class FadeMovingAverage_Random(Pertubation):
    def __init__(self, device, eps=1.0e-7, alpha_init=0.9):
        super().__init__(device, eps)
        self.alpha = torch.tensor(alpha_init, requires_grad=True, device=device)
    
    def apply(self, X, mask_tensor, top_k):
        '''
        X: [batch_size, channels, num_nodes, time_steps]
        mask_tensor: [batch_size, channels, num_nodes, time_steps]
        top_k: number of time steps to select randomly
        '''
        super().apply(X, mask_tensor)
        # Ensure mask_tensor has the correct shape along the num_nodes dimension
        mask_tensor = self.adjust_mask_tensor(mask_tensor, X.size(2))
        # Calculate the moving average across time steps
        moving_average = torch.mean(X, dim=-1, keepdim=True)  # [batch_size, channels, num_nodes, 1]
        moving_average_tilted = moving_average.expand_as(X)  # [batch_size, channels, num_nodes, time_steps]
        # Expand mask_tensor to match X's shape
        mask_tensor_expanded = mask_tensor.expand_as(X)
        # Apply perturbation
        X_pert = mask_tensor_expanded * X + (1 - mask_tensor_expanded) * moving_average_tilted
        random_indices = torch.randint(0, X.size(-1), (top_k,), device=self.device)
        random_indices_sorted = torch.sort(random_indices).indices
        random_mask = torch.zeros(X.size(-1), device=self.device, dtype=torch.bool)
        random_mask[random_indices_sorted] = True
        random_mask_expanded = random_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(X)
        X_pert_random = torch.where(random_mask_expanded, X_pert, torch.zeros_like(X))
        return X_pert_random, random_indices_sorted
    
    
    
class ModelWrapper(nn.Module):
    def __init__(self, original_model):
        super(ModelWrapper, self).__init__()
        self.original_model = original_model
        # self.adj= adj
    def forward(self, x):
        _, _, adj = util.load_adj('store/METR-LA/adj_mx.pkl')
        edge_index = [[], []]
        edge_weight = []

        for i in range(207):
            for j in range(207):
                if adj.item((i, j)) != 0:
                    edge_index[0].append(i)
                    edge_index[1].append(j)
                    edge_weight.append(adj.item((i, j)))

        adj_mx = torch.tensor(adj).cuda()
        output = self.original_model(x, adj_mx)  # Shape: (batch_size, seq_length, num_nodes, time_steps)
        reshaped_output = output[:, :, :, -1].view(output.size(0), -1)  # Shape: (batch_size, seq_length * num_nodes)
        return reshaped_output


class MSTEWrapper(nn.Module):
    def __init__(self, model: nn.Module, adj_mx: torch.Tensor):
        super(MSTEWrapper, self).__init__()
        self.model = model
        self.adj_mx = adj_mx

    def forward(self, backcast: torch.Tensor):
        return self.model(backcast, self.adj_mx)
    
    
def calculate_DeepLift(model, X, adj, top_k, status):
    model_wrapper = MSTEWrapper(model, adj).cuda()
    target = target = (0, 0)
    deeplift = DeepLift(model_wrapper)
    attributions, _ = deeplift.attribute(X, target=target, return_convergence_delta=True)
    top_k_values, top_k_indices = torch.topk(attributions, top_k, dim=2, largest=status, sorted=False)
    mask = torch.zeros_like(attributions, dtype=torch.bool)
    mask.scatter_(2, top_k_indices, True)
    top_k_M = attributions * mask
    epsilon = 1e-6
    top_k_M = torch.where(torch.abs(top_k_M) < epsilon, torch.tensor(0.0), top_k_M)
    return top_k_M


def calculate_GradientShap(model, X, adj, top_k, status):
    model_wrapper = MSTEWrapper(model, adj).cuda()
    target = target = (0, 0)
    gradientShap = GradientShap(model_wrapper)
    baselines = torch.zeros_like(X)
    attributions, _ = gradientShap.attribute(X, baselines, target=target, return_convergence_delta=True)
    top_k_values, top_k_indices = torch.topk(attributions, top_k, dim=2, largest=status, sorted=False)
    mask = torch.zeros_like(attributions, dtype=torch.bool)
    mask.scatter_(2, top_k_indices, True)
    top_k_M = attributions * mask
    epsilon = 1e-6
    top_k_M = torch.where(torch.abs(top_k_M) < epsilon, torch.tensor(0.0), top_k_M)
    return top_k_M


def calculate_DeepLiftShap(model, X, adj, top_k, status):
    model_wrapper = MSTEWrapper(model, adj).cuda()
    deeplift_shap = DeepLiftShap(model_wrapper)
    target = target = (0, 0)
    baseline = torch.zeros_like(X).cuda()
    baselines = baseline.repeat(X.size(0), 1, 1, 1)
    # Calculate the attributions
    attributions = deeplift_shap.attribute(X, baselines=baselines, target=target)
    # Assume the `target` is the class index you are interested in, or None for all outputs
    # Select the top-k attributions
    top_k_values, top_k_indices = torch.topk(attributions, top_k, dim=2, largest=status, sorted=False)
    mask = torch.zeros_like(attributions, dtype=torch.bool)
    mask.scatter_(2, top_k_indices, True)
    top_k_M = attributions * mask
    epsilon = 1e-6
    top_k_M = torch.where(torch.abs(top_k_M) < epsilon, torch.tensor(0.0).cuda(), top_k_M)
    return top_k_M


def calculate_Lime(model, X, adj, top_k, status):
    model_wrapper = MSTEWrapper(model, adj).cuda()
    lime = Lime(model_wrapper)
    target = (0, 0)
    top_k_M_all = torch.zeros_like(X)
    for i in range(X.shape[0]):
        explanation = lime.attribute(X[i:i+1], target=target)
        top_k_values, top_k_indices = torch.topk(explanation, top_k, dim=2, largest=status, sorted=False)
        mask = torch.zeros_like(explanation, dtype=torch.bool)
        mask.scatter_(2, top_k_indices, True)
        top_k_M = explanation * mask
        epsilon = 1e-6
        top_k_M = torch.where(torch.abs(top_k_M) < epsilon, torch.tensor(0.0).cuda(), top_k_M)
        top_k_M_all[i:i+1] = top_k_M
    return top_k_M_all


class MSTEWrapperShap(nn.Module):
    def __init__(self, model: nn.Module, adj_mx: torch.Tensor, output_index=None):
        super(MSTEWrapperShap, self).__init__()
        self.model = model
        self.adj_mx = adj_mx
        self.output_index = output_index

    def forward(self, backcast: torch.Tensor):
        output = self.model(backcast, self.adj_mx)
        if self.output_index is not None:
            output = output[:, self.output_index]
        return output


def calculate_Shap(model, X, adj, top_k, status, output_index=0):
    model_wrapper = MSTEWrapperShap(model, adj, output_index=output_index).cuda()
    explainer = shap.GradientExplainer(model_wrapper, X)
    shap_values = explainer.shap_values(X)
    top_k_values, top_k_indices = torch.topk(shap_values, top_k, dim=2, largest=status, sorted=False)
    mask = torch.zeros_like(shap_values, dtype=torch.bool)
    mask.scatter_(2, top_k_indices, True)
    top_k_M = shap_values * mask
    epsilon = 1e-6
    top_k_M = torch.where(torch.abs(top_k_M) < epsilon, torch.tensor(0.0), top_k_M)
    return top_k_M


class FadeMovingAverage_GRU(Pertubation):
    def __init__(self, device, eps=1.0e-7, alpha_init=0.9):
        super().__init__(device, eps)
        self.alpha = torch.tensor(alpha_init, requires_grad=True, device=device)
        self.gru = nn.GRU(input_size=2, hidden_size=2, batch_first=True).to(device)
    
    def apply(self, X, mask_tensor, top_k, status):
        '''
        X: [batch_size, channels, num_nodes, time_steps]
        mask_tensor: [batch_size, channels, num_nodes, time_steps]
        top_k: number of top time steps to select based on influence
        '''
        super().apply(X, mask_tensor)
        mask_tensor = self.adjust_mask_tensor(mask_tensor, X.size(2))
        # Reshape X for GRU input: [batch_size * num_nodes, time_steps, channels]
        batch_size, channels, num_nodes, time_steps = X.size()
        X_reshaped = X.permute(0, 2, 3, 1).reshape(batch_size * num_nodes, time_steps, channels)  # [batch_size * num_nodes, time_steps, channels]
        # Pass through GRU
        gru_out, _ = self.gru(X_reshaped)  # [batch_size * num_nodes, time_steps, hidden_size]
        gru_out = gru_out[:, -1, :]  # Take the output of the last time step [batch_size * num_nodes, hidden_size]
        gru_out = gru_out.reshape(batch_size, num_nodes, -1)  # [batch_size, num_nodes, hidden_size]
        gru_out = gru_out.permute(0, 2, 1)  # [batch_size, hidden_size, num_nodes]
        # Expand the GRU output to match the original time steps
        gru_out_expanded = gru_out.unsqueeze(3).expand(-1, -1, -1, time_steps)  # [batch_size, hidden_size, num_nodes, time_steps]
        # Expand mask_tensor to match X's shape
        mask_tensor_expanded = mask_tensor.expand_as(X)
        # Apply perturbation
        X_pert = mask_tensor_expanded * X + (1 - mask_tensor_expanded) * gru_out_expanded
        # Calculate the influence scores for each time step
        influence_scores = torch.abs(X - gru_out_expanded)  # [batch_size, channels, num_nodes, time_steps]
        influence_scores = torch.mean(influence_scores, dim=(0, 1, 2)) 
        _, top_indices = torch.topk(influence_scores, k=top_k, largest=status)
        top_k_mask = torch.zeros_like(X, dtype=torch.bool)
        top_k_mask[:, :, :, top_indices] = True
        X_pert_top_k = torch.where(top_k_mask, X_pert, torch.zeros_like(X))
        return X_pert_top_k, top_indices

    

