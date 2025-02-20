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

def stgode_adj_load(dataset, sigma1=0.1, sigma2=10, threshold1=0.6, threshold2=0.5):
    from fastdtw import fastdtw
    
    infos = {
        'METR-LA': {'filetype': 'h5'},
        'PEMS-BAY': {'filetype': 'h5'},
        'PEMS04': {'filetype': 'npz'},
        'PEMS08': {'filetype': 'npz'},
        'PEMS03': {'filetype': 'npz'},
        'PEMS07': {'filetype': 'npz'},
    }
    filetype = infos[dataset]['filetype']
    
    raw_path = './store/raw/{}'.format(dataset)
    processed_path = './store/{}'.format(dataset)
    
    datafile_path = raw_path + '/data.{}'.format(filetype)
    
    dtw_path = processed_path + '/dtw_distance.npy'
    spatial_path = processed_path + '/adj_mx.pkl'
    
    if not os.path.exists(dtw_path) or not os.path.exists(spatial_path):
        if filetype == 'h5':
            data = pd.read_hdf(datafile_path).values
            data = np.expand_dims(data, axis=-1)
        elif filetype == 'npz':
            data = np.load(datafile_path)['data']
        
        # data: [n_samples, num_nodes, channels]
        num_nodes = data.shape[1]
        mean_value = np.mean(data, axis=(0, 1)).reshape(1, 1, -1)
        std_value = np.std(data, axis=(0, 1)).reshape(1, 1, -1)
        data = (data - mean_value) / std_value
        mean_value = mean_value.reshape(-1)[0]
        std_value = std_value.reshape(-1)[0]
        
        # generate distance matrix
        if not os.path.exists(dtw_path):
            data_mean = np.mean([data[:, :, 0][24*12*i: 24*12*(i+1)] for i in range(data.shape[0]//(24*12))], axis=0)
            data_mean = data_mean.squeeze().T 
            dtw_distance = np.zeros((num_nodes, num_nodes))
            for i in range(num_nodes):
                for j in range(i, num_nodes):
                    dtw_distance[i][j] = fastdtw(data_mean[i], data_mean[j], radius=6)[0]
            for i in range(num_nodes):
                for j in range(i):
                    dtw_distance[i][j] = dtw_distance[j][i]
            np.save(dtw_path, dtw_distance)
    
    # loading distance matrix
    dist_matrix = np.load(dtw_path)
    # normalization
    mean = np.mean(dist_matrix)
    std = np.std(dist_matrix)
    dist_matrix = (dist_matrix - mean) / std
    sigma = sigma1
    dist_matrix = np.exp(-dist_matrix ** 2 / sigma ** 2)
    dtw_matrix = np.zeros_like(dist_matrix)
    dtw_matrix[dist_matrix > threshold1] = 1
    
    # loading spatial matrix
    try:
        _, _, dist_matrix = load_adj(spatial_path)
    except:
        dist_matrix = load_adj(spatial_path)
    # normalization
    std = np.std(dist_matrix[dist_matrix != np.float64('inf')])
    mean = np.mean(dist_matrix[dist_matrix != np.float64('inf')])
    dist_matrix = (dist_matrix - mean) / std
    sigma = sigma2
    sp_matrix = np.exp(- dist_matrix**2 / sigma**2)
    sp_matrix[sp_matrix < threshold2] = 0 
    
    return dtw_matrix, sp_matrix

def stgode_normalized_adj(A):
    """
    Returns a tensor, the degree normalized adjacency matrix.
    """
    alpha = 0.8
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A), diag.reshape((1, -1)))
    A_reg = alpha / 2 * (np.eye(A.shape[0]) + A_wave)
    return torch.from_numpy(A_reg.astype(np.float32))

def gman_seq2instance(data, num_his=12, num_pred=12):
    num_step, dims = data.shape
    num_sample = num_step - num_his - num_pred + 1
    x = torch.zeros(num_sample, num_his, dims)
    y = torch.zeros(num_sample, num_pred, dims)
    for i in range(num_sample):
        x[i] = data[i: i + num_his]
        y[i] = data[i + num_his: i + num_his + num_pred]
    return x, y

def gman_data_preprocess(dataloader, 
                         num_nodes: int, 
                         steps_per_day=12 * 24,
                         K=8,
                         d=8,
                         add_time_of_day=True, 
                         add_day_of_week=True,
                         num_his=12,
                         num_pred=12):
    '''
        only call this function for PEMS04 & PEMS08
        TODO: rewrite to use time values in METR-LA and PEMS-BAY
    '''
    t = num_his + num_pred
    train = (dataloader['train_loader'].num_batch, dataloader['train_batch_size'])
    val = (dataloader['val_loader'].num_batch, dataloader['val_batch_size'])
    test = (dataloader['test_loader'].num_batch, dataloader['test_batch_size'])
    
    train_steps = t + (train[0] * train[1] - 1)
    val_steps = t + (val[0] * val[1] - 1)
    test_steps = t + (test[0] * test[1] - 1)
    
    n_samples = train_steps + val_steps + test_steps
    
    if add_time_of_day:
        # numerical time_of_day
        tod = [i % steps_per_day / steps_per_day for i in range(n_samples)]
        # tod = np.array(tod)
        # tod_tiled = np.tile(tod, [1, num_nodes, 1]).transpose((2, 1, 0))
        tod = torch.tensor(tod).unsqueeze(-1)
    
    if add_day_of_week:
        # numerical day_of_week
        dow = [(i // steps_per_day) % 7 for i in range(n_samples)]
        # dow = np.array(dow)
        # dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        dow = torch.tensor(dow).unsqueeze(-1)
    
    time = torch.cat((dow, tod), dim=-1)
    
    train = time[: train_steps]
    val = time[train_steps: train_steps + val_steps]
    test = time[-test_steps:]
    
    trainTE = gman_seq2instance(train, num_his, num_pred)
    trainTE = torch.cat(trainTE, 1).type(torch.int32)
    
    valTE = gman_seq2instance(val, num_his, num_pred)
    valTE = torch.cat(valTE, 1).type(torch.int32)
    
    testTE = gman_seq2instance(test, num_his, num_pred)
    testTE = torch.cat(testTE, 1).type(torch.int32)
    
    # add padding
    # batch_size = dataloader['train_batch_size']
    # num_padding = (batch_size - (len(dataloader['x_train']) % batch_size)) % batch_size
    # trainTE_padding = np.repeat(trainTE[-1:], num_padding, axis=0)
    # trainTE = np.concatenate([trainTE, trainTE_padding], axis=0)
    # trainTE = torch.tensor(trainTE)
    
    # batch_size = dataloader['val_batch_size']
    # num_padding = (batch_size - (len(dataloader['x_val']) % batch_size)) % batch_size
    # valTE_padding = np.repeat(valTE[-1:], num_padding, axis=0)
    # valTE = np.concatenate([valTE, valTE_padding], axis=0)
    # valTE = torch.tensor(valTE)
    
    # batch_size = dataloader['test_batch_size']
    # num_padding = (batch_size - (len(dataloader['x_test']) % batch_size)) % batch_size
    # testTE_padding = np.repeat(testTE[-1:], num_padding, axis=0)
    # testTE = np.concatenate([testTE, testTE_padding], axis=0)
    # testTE = torch.tensor(testTE)
    
    # SE = torch.rand((num_nodes, K * d))
    SE = torch.zeros((num_nodes, K * d))
    
    return SE, trainTE, valTE, testTE

def stgcn_calc_gso(dir_adj, gso_type):
    n_vertex = dir_adj.shape[0]

    if sp.issparse(dir_adj) == False:
        dir_adj = sp.csc_matrix(dir_adj)
    elif dir_adj.format != 'csc':
        dir_adj = dir_adj.tocsc()
    
    id = sp.identity(n_vertex, format='csc')
    
    # Symmetrizing an adjacency matrix
    adj = dir_adj + dir_adj.T.multiply(dir_adj.T > dir_adj) - dir_adj.multiply(dir_adj.T > dir_adj)
    #adj = 0.5 * (dir_adj + dir_adj.transpose())
    
    if gso_type == 'sym_renorm_adj' or gso_type == 'rw_renorm_adj' \
        or gso_type == 'sym_renorm_lap' or gso_type == 'rw_renorm_lap':
        adj = adj + id
    
    if gso_type == 'sym_norm_adj' or gso_type == 'sym_renorm_adj' \
        or gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
        row_sum = adj.sum(axis=1).A1
        row_sum_inv_sqrt = np.power(row_sum, -0.5)
        row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.
        deg_inv_sqrt = sp.diags(row_sum_inv_sqrt, format='csc')
        # A_{sym} = D^{-0.5} * A * D^{-0.5}
        sym_norm_adj = deg_inv_sqrt.dot(adj).dot(deg_inv_sqrt)

        if gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
            sym_norm_lap = id - sym_norm_adj
            gso = sym_norm_lap
        else:
            gso = sym_norm_adj

    elif gso_type == 'rw_norm_adj' or gso_type == 'rw_renorm_adj' \
        or gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
        row_sum = np.sum(adj, axis=1).A1
        row_sum_inv = np.power(row_sum, -1)
        row_sum_inv[np.isinf(row_sum_inv)] = 0.
        deg_inv = np.diag(row_sum_inv)
        # A_{rw} = D^{-1} * A
        rw_norm_adj = deg_inv.dot(adj)

        if gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
            rw_norm_lap = id - rw_norm_adj
            gso = rw_norm_lap
        else:
            gso = rw_norm_adj

    else:
        raise ValueError(f'{gso_type} is not defined.')

    return gso

def stgcn_calc_chebynet_gso(gso):
    if sp.issparse(gso) == False:
        gso = sp.csc_matrix(gso)
    elif gso.format != 'csc':
        gso = gso.tocsc()

    id = sp.identity(gso.shape[0], format='csc')
    # If you encounter a NotImplementedError, please update your scipy version to 1.10.1 or later.
    eigval_max = norm(gso, 2)

    # If the gso is symmetric or random walk normalized Laplacian,
    # then the maximum eigenvalue is smaller than or equals to 2.
    if eigval_max >= 2:
        gso = gso - id
    else:
        gso = 2 * gso / eigval_max - id

    return gso

def calculate_model_size(model: torch.nn.Module):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_in_mb = (param_size + buffer_size) / 1024**2
    return size_in_mb