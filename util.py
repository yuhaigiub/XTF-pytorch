import os
import pickle

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from scipy.sparse.linalg import norm

class DataLoader(object):
    """
    This class loads the train, val and test data to be fed to the model
    """
    def __init__(self, xs, ys, batch_size):
        self.batch_size = batch_size
        self.current_ind = 0
        num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
        x_padding = np.repeat(xs[-1:], num_padding, axis=0)
        y_padding = np.repeat(ys[-1:], num_padding, axis=0)
        xs = np.concatenate([xs, x_padding], axis=0)
        ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        
        self.xs = xs
        self.ys = ys
    
    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys
    
    def get_iterator(self):
        self.current_ind = 0
    
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size,
                              self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1
        
        return _wrapper()

class StandardScaler():
    """
    Standardize the input
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def transform(self, data: torch.Tensor, i=0):
        return (data - self.mean[i]) / self.std[i]
    
    def inverse_transform(self, data: torch.Tensor, i=0):
        return (data * self.std[i]) + self.mean[i]

def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(pkl_filename):
    try:
        _, _, adj_mx = load_pickle(pkl_filename)
    except:
        adj_mx = load_pickle(pkl_filename)
    
    # remove self loop
    np.fill_diagonal(adj_mx, 0.0)
    
    return adj_mx

def load_dataset(dataset_dir, 
                 batch_size, 
                 valid_batch_size=None, 
                 test_batch_size=None, 
                 indices=(0, )):
    if valid_batch_size is None: valid_batch_size = batch_size
    if test_batch_size is None: test_batch_size = batch_size
    
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    
    mean = np.mean(data['x_train'], axis=(0, 1, 2))
    std = np.std(data['x_train'], axis=(0, 1, 2))
    
    scaler = StandardScaler(mean, std)
    
    for category in ['train', 'val', 'test']:
        for i in indices:
            data['x_' + category][..., i] = scaler.transform(data['x_' + category][..., i], i)
    
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    
    data['train_batch_size'] = batch_size
    data['val_batch_size'] = valid_batch_size
    data['test_batch_size'] = test_batch_size
    
    return data

def masked_mse(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=0.0):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))

def masked_mae(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mape(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_huber(preds, labels, delta=1.0, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = F.huber_loss(preds, labels, reduction='none', delta=delta)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse

def calculate_model_size(model: torch.nn.Module):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_in_mb = (param_size + buffer_size) / 1024**2
    return size_in_mb