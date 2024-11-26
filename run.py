import argparse
import os
import time

import numpy as np
import torch
import torch.optim as optim

import util

from model import MSTE

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='METR-LA')
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--start', type=int, default=-1, help='using for save pth file')
    
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--wdecay', type=float, default=0.0001, help='weight decay rate')
    parser.add_argument('--w2', type=float, default=0.001)
    parser.add_argument('--w3', type=float, default=0.0001)
    
    parser.add_argument('--print_every', type=int, default=100, help='')
    
    parser.add_argument('--in_dim', type=int, default=2)
    parser.add_argument('--conv_dim', type=int, default=32)
    parser.add_argument('--end_dim', type=int, default=64)
    parser.add_argument('--seq_len', type=int, default=12, help='time step')
    args = parser.parse_args()
    
    args.model_name = 'mste'
    args.data = './store/{}'.format(args.dataset)
    args.adjdata = './store/{}/adj_mx.pkl'.format(args.dataset)
    args.save = './saved_models/{}/{}'.format(args.dataset, args.model_name)
    
    return args

def main(args):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda') # force cuda
    
    try:
        _, _, adj_mx = util.load_adj(args.adjdata)
    except:
        adj_mx = util.load_adj(args.adjdata)
    
    adj_mx = torch.tensor(adj_mx).to(device)
    dataloader = util.load_dataset(args.data, args.batch_size)
    scaler = dataloader['scaler']
    
    model = MSTE(n_experts=3, n_stacks=1,
                 in_dim=args.in_dim, seq_len=args.seq_len,
                 conv_dim=args.conv_dim, end_dim=args.end_dim,
                 adj_mx=adj_mx,
                 time_0=0.9, step_size_0=0.45,
                 time_1=0.9, step_size_1=0.9,
                 time_2=0.9, step_size_2=0.3,)
    model.to(device)
    
    # load checkpoint
    if args.epochs != 0 and args.start != -1:
        print('loading epoch {}'.format(args.start))
        model.load_state_dict(torch.load(args.save + '/G_T_model_{}.pth'.format(args.start)))
    
    engine = Engine(scaler, model, args.lr, args.wdecay, args.w2, args.w3, device)
    
    schedulerLR = None
    # milestone = [5, 10, 15]
    # schedulerLR = optim.lr_scheduler.MultiStepLR(engine.optimizer, milestones=milestone, gamma=0.1)
    
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    
    log_file_train = open('loss_train_log.txt', 'w')
    log_file_val = open('loss_val_log.txt', 'w')
    log_file_test = open('loss_test_log.txt', 'w')
    
    print("start training...", flush=True)
    
    his_loss = []
    val_time = []
    train_time = []
    best_epoch = 0 if args.epochs != 0 else args.start
    
    for epoch in range(args.start + 1, args.start + 1 + args.epochs):
        print(f'----------Training epoch {epoch:03d}----------')
        train_loss = []
        train_mape = []
        train_rmse = []
        
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        
        for i, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            # [B, L, N, C]
            trainx = torch.Tensor(x).to(device)
            trainy = torch.Tensor(y).to(device)
            
            metrics = engine.train(trainx, trainy[..., 0:1])
            
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            
            if i % args.print_every == 0:
                log = 'Iter: {:03d}, Train(Loss: {:.4f}, MAPE: ' + '{:.4f}, RMSE: {:.4f}), l2: {:.2f}, l3: {:.2f}'
                print(log.format(i, train_loss[-1], train_mape[-1], train_rmse[-1], metrics[-2], metrics[-1]), flush=True)
        
        if schedulerLR is not None:
            schedulerLR.step()
        
        t2 = time.time()
        train_time.append(t2 - t1)
        print(f'finish train loop. Time: {t2-t1:.2f}s')
        
        valid_loss = []
        valid_mape = []
        valid_rmse = []
        
        s1 = time.time()
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
            
            log = 'Epoch {:03d}, Loss: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f} \n'
            log_file_train.write(log.format(epoch, mtrain_loss, mtrain_mape, mtrain_rmse))
            log_file_train.flush()
            log_file_val.write(log.format(epoch, mvalid_loss, mvalid_mape, mvalid_rmse))
            log_file_val.flush()
            log_file_test.write(log.format(epoch, mtest_loss, mtest_mape, mtest_rmse))
            log_file_test.flush()

class Engine():
    def __init__(self, 
                 scaler, 
                 model,
                 lrate, wdecay, w2, w3,
                 device,
                 huber_delta=1.0,):
        self.model = model
        self.model.to(device)
        
        self.lrate = lrate
        self.wdecay = wdecay
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        
        print('number of parameters:', sum(p.numel() for p in model.parameters()))
        
        self.loss = lambda o, l: util.masked_huber(o, l, huber_delta)
        self.scaler = scaler
        self.clip = 3.0
        self.w2 = w2
        self.w3 = w3
    
    def train(self, input, real):
        '''
        input: [B, L, N, C]
        '''
        self.model.train()
        
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
                # Loop through the parameters of both models and compute the L2 norm
                for param1, param2 in zip(model1.parameters(), model2.parameters()):
                    diff = param1 - param2  # Compute difference between weights
                    l2_norm = l2_norm + torch.norm(diff) ** 2  # Accumulate squared norms
                
                # Take the square root of the accumulated squared norms
                l3 = l3 + torch.sqrt(l2_norm)
        
        loss = l1 + self.w2 * l2 + self.w3 * l3
        loss.backward()
        
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        
        mape = util.masked_mape(predict, real)
        rmse = util.masked_rmse(predict, real)
        return loss.item(), mape.item(), rmse.item(), l2.item() * self.w2, l3.item() * self.w3
    
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

if __name__ == "__main__":
    t1 = time.time()
    args = parse_args()
    main(args)
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))