import argparse
import time

import numpy as np
import torch

import util

from model import MSTE

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='METR-LA')
    
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--start', type=int, default=-1, help='using for save pth file')
        
    parser.add_argument('--in_dim', type=int, default=2)
    parser.add_argument('--seq_len', type=int, default=12, help='time step')
    parser.add_argument('--conv_dim', type=int, default=32)
    parser.add_argument('--end_dim', type=int, default=64)
    
    parser.add_argument('--n_experts', type=int, default=3)
    parser.add_argument('--n_stacks', type=int, default=3)
    # number of blocks
    parser.add_argument('--time_0', type=float, default=0.9)
    parser.add_argument('--step_0', type=float, default=0.9)
    # number of temporal ode steps
    parser.add_argument('--time_1', type=float, default=0.9)
    parser.add_argument('--step_1', type=float, default=0.3)
    # number of spatial ode steps
    parser.add_argument('--time_2', type=float, default=0.9)
    parser.add_argument('--step_2', type=float, default=0.3)
    
    parser.add_argument('--dropout', type=float, default=0.5)
    
    parser.add_argument('--decoder_types', type=str, default='1,1',
                        help='number separated by comma (no whitespace allowed)')
    
    args = parser.parse_args()
    
    args.model_name = 'mste'
    args.data = './store/{}'.format(args.dataset)
    args.adjdata = './store/{}/adj_mx.pkl'.format(args.dataset)
    args.save = './saved_models/{}/{}'.format(args.dataset, args.model_name)
    
    return args

def test(args):
    print('Horizon Test')
    
    device = torch.device(args.device)
    
    try:
        _, _, adj_mx = util.load_adj(args.adjdata)
    except:
        adj_mx = util.load_adj(args.adjdata)
    
    adj_mx = torch.tensor(adj_mx).to(device)
    dataloader = util.load_dataset(args.data, args.batch_size)
    scaler = dataloader['scaler']
    
    decoder_types = [int(x) for x in args.decoder_types.split(',')]
    print('decoder types:', decoder_types)
    
    model = MSTE(args.n_experts, args.n_stacks,
                 args.in_dim, args.seq_len,
                 args.conv_dim, args.end_dim,
                 adj_mx,
                 args.time_0, args.step_0,
                 args.time_1, args.step_1,
                 args.time_2, args.step_2,
                 decoder_types,
                 args.dropout)
    model.to(device)
    
    # load checkpoint
    if args.start != -1:
        print('loading epoch {}'.format(args.start))
        model.load_state_dict(torch.load(args.save + '/G_T_model_{}.pth'.format(args.start), weights_only=True))
    
    print('number of parameters:', sum(p.numel() for p in model.parameters()))
    
    # horizon test
    with torch.no_grad():
        model.eval()
        outputs = []
        realy = torch.Tensor(dataloader['y_test']).to(device)
        realy = realy[..., 0] # [B, L, N]
        
        for i, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            out = model(testx)[..., 0] # [B, L, N]
            outputs.append(out)
        
        yhat = torch.cat(outputs, dim=0)
        yhat = yhat[:realy.size(0), ...]
        
        amae = []
        amape = []
        armse = []
        for i in range(12):
            pred = scaler.inverse_transform(yhat[:, i, :])
            real = realy[:, i, :]
            
            metrics = util.metric(pred, real)
            
            log = 'horizon {:d}, MAE:' + '{:.4f}, MAPE: {:.4f}, RMSE: {:.4f}'
            print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
            amae.append(metrics[0])
            amape.append(metrics[1])
            armse.append(metrics[2])
    
    mtest_loss = np.mean(amae)
    mtest_mape = np.mean(amape)
    mtest_rmse = np.mean(armse)
    log = ('On average over 12 horizons, MAE: {:.4f}, MAPE: ' + '{:.4f}, RMSE: {:.4f}')
    print(log.format(mtest_loss, mtest_mape, mtest_rmse))

if __name__ == "__main__":
    t1 = time.time()
    args = parse_args()
    test(args)
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))