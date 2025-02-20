import argparse
import os
import time

import numpy as np
import torch

import util

from model import MSTE
from engine import Engine

torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='METR-LA')
    
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--start', type=int, default=-1, help='using for save pth file')
    
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--wdecay', type=float, default=0.0001, help='weight decay rate')
    parser.add_argument('--clip', type=float, default=3)
    parser.add_argument('--w2', type=float, default=0.001)
    parser.add_argument('--w3', type=float, default=0.0001)
    parser.add_argument('--milestones', type=str, default=None, 
                        help='number separated by comma (no whitespace allowed)')
    parser.add_argument('--grad_acc_step_max', type=int, default=1)
    parser.add_argument('--huber_delta', type=str, default=1.0)
    
    parser.add_argument('--print_every', type=int, default=100, help='')
    
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

def main(args):
    device = torch.device(args.device)
    
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    
    log_file_train = open('loss_train_log.txt', 'w')
    log_file_val = open('loss_val_log.txt', 'w')
    log_file_test = open('loss_test_log.txt', 'w')
    
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
    
    engine = Engine(scaler, 
                    model, 
                    args.lr, args.wdecay,
                    args.w2, args.w3, 
                    device, 
                    args.grad_acc_step_max,
                    args.clip,
                    args.huber_delta)
    
    if args.milestones is None:
        schedulerLR = None
    else:
        milestones = [int(x) for x in args.milestones.split(",")]
        schedulerLR = torch.optim.lr_scheduler.MultiStepLR(engine.optimizer, milestones, 0.1, last_epoch=args.start)
        print(milestones)
    
    print("start training...", flush=True)
    
    his_loss = []
    val_time = []
    train_time = []
    best_epoch = max(0, args.start)
    
    engine.optimizer.zero_grad()
    
    for epoch in range(args.start + 1, args.start + 1 + args.epochs):
        print(f'----------Training epoch {epoch:03d}----------')
        train_loss = []
        train_mape = []
        train_rmse = []
        
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        
        t_cp = time.time()
        for i, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            # [B, L, N, C]
            trainx = torch.Tensor(x).to(device)
            trainy = torch.Tensor(y).to(device)
            
            metrics = engine.train(trainx, trainy[..., 0:1])
            
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            
            if i % args.print_every == 0:
                t_cp_2 = time.time()
                log = 'Iter: {:03d}, Train(Loss: {:.4f}, MAPE: ' + '{:.4f}, RMSE: {:.4f}), l2: {:.2f}, l3: {:.2f}, t={:.2f}'
                print(log.format(i, 
                                 train_loss[-1], train_mape[-1], train_rmse[-1], 
                                 metrics[-2], metrics[-1], 
                                 t_cp_2 - t_cp), 
                      flush=True)
                t_cp = t_cp_2
        
        if schedulerLR is not None:
            schedulerLR.step()
        
        t2 = time.time()
        train_time.append(t2 - t1)
        print(f'finish train loop. Time: {t2-t1:.2f}s')
        
        valid_loss = []
        valid_mape = []
        valid_rmse = []
        
        s1 = time.time()
        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
                valx = torch.Tensor(x).to(device)
                valy = torch.Tensor(y).to(device)
                
                metrics = engine.eval(valx, valy[..., 0:1])
                valid_loss.append(metrics[0])
                valid_mape.append(metrics[1])
                valid_rmse.append(metrics[2])
        
        s2 = time.time()
        print(f'finish val loop. Time: {s2-s1:.2f}s')
        val_time.append(s2 - s1)
        
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        
        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        
        log = 'Epoch: {:03d}\n' + \
              'Train (Loss: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}), Time: {:.4f}s\n' + \
              'Valid (Loss: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}), Time: {:.4f}s\n'
        
        print(log.format(epoch, 
                         mtrain_loss, mtrain_mape, mtrain_rmse, (t2 - t1),
                         mvalid_loss, mvalid_mape, mvalid_rmse, (s2 - s1)))
        
        # save model
        torch.save(engine.model.state_dict(), args.save + "/G_T_model_" + str(epoch) + ".pth")
        
        his_loss.append(mvalid_loss)
        if np.argmin(his_loss) == len(his_loss) - 1:
            # only save good epochs
            best_epoch = epoch
        
        # horizon test
        z1 = time.time()
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
        
        z2 = time.time()
        
        mtest_loss = np.mean(amae)
        mtest_mape = np.mean(amape)
        mtest_rmse = np.mean(armse)
        log = ('On average over 12 horizons, MAE: {:.4f}, MAPE: ' + '{:.4f}, RMSE: {:.4f}')
        print(log.format(mtest_loss, mtest_mape, mtest_rmse))
        print(f'time spend: {z2 - z1:.2f}')
        log = 'Epoch {:03d}, Loss: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f} \n'
        log_file_train.write(log.format(epoch, mtrain_loss, mtrain_mape, mtrain_rmse))
        log_file_train.flush()
        log_file_val.write(log.format(epoch, mvalid_loss, mvalid_mape, mvalid_rmse))
        log_file_val.flush()
        log_file_test.write(log.format(epoch, mtest_loss, mtest_mape, mtest_rmse))
        log_file_test.flush()

if __name__ == "__main__":
    t1 = time.time()
    args = parse_args()
    main(args)
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))