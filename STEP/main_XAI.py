import argparse
import os
import time
import numpy as np
import torch
import util
from pertubate import FadeMovingAverage, FadeMovingAverage_TopK, FadeMovingAverage_Random, perturbate_node, degreeCentrality, closenessCentrality
from engine_XAI import EngineXAI
from model import MSTE

parser = argparse.ArgumentParser()


parser.add_argument('--device', type=str, default='cuda', help='device to run the model on')
parser.add_argument('--data', type=str, default='./store/METR-LA', help='data path')
parser.add_argument('--adjdata', type=str, default='./store/METR-LA/adj_mx.pkl', help='adj data path')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--print_every', type=int, default=50, help='')
parser.add_argument('--save', type=str, default='saved_models/XAI/GWNET', help='save path')
parser.add_argument('--blackbox_file', 
                    type=str, 
                    default='save_blackbox/G_T_model_0.pth', 
                    help='blackbox .pth file')
parser.add_argument('--num_nodes', type=int, default=207, help='number of nodes')
parser.add_argument('--seq_len', type=int, default=12, help='time step')
parser.add_argument('--in_dim', type=int, default=2, help='in channels')
parser.add_argument('--out_dim', type=int, default=1, help='out channels')
parser.add_argument('--batch_size', type=int, default=16, help='settings')
parser.add_argument('--epochs', type=int, default=1, help='settings')
parser.add_argument('--iter_epoch', type=int, default=-1, help='using for save pth file')
parser.add_argument('--single_test', type=bool, default=True, help='settings')
parser.add_argument('--n_cells', type=int, default=3, help='number of ensemble cells')
parser.add_argument('--conv_dim', type=int, default=32, help='settings')
parser.add_argument('--end_dim', type=int, default=64, help='settings')
parser.add_argument('--train_loss', type=str, default='mae', help='settings')
parser.add_argument('--test_loss', type=str, default='mae', help='settings')
parser.add_argument('--huber_delta', type=float, default=1.0)
parser.add_argument('--setting_id', type=str, default="Int")
parser.add_argument('--combine_method', type=str, default='mean', choices=['sum', 'mean'])
parser.add_argument('--decoder_method', type=str, default='1', choices=['1', '2'])

DATASET_CHOICES = ['METR-LA', 'PEMS-BAY', 'PEMS03', 'PEMS04', 'PEMS07', 'PEMS08']
parser.add_argument('--dataset', type=str, default='METR-LA', 
                    choices=DATASET_CHOICES)


args = parser.parse_args()

def main():
    device = torch.device(args.device)
    _, _, adj_mx = util.load_adj(args.adjdata)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    pertubate = FadeMovingAverage_TopK(device)
    
    model = MSTE(n_experts=3, n_stacks=1,
                 in_dim=args.in_dim, seq_len=args.seq_len,
                 conv_dim=args.conv_dim, end_dim=args.end_dim,
                 adj_mx=adj_mx,
                 time_0=0.9, step_size_0=0.45,
                 time_1=0.9, step_size_1=0.9,
                 time_2=0.9, step_size_2=0.3,)

    blackbox =  MSTE(n_experts=3, n_stacks=1,
                 in_dim=args.in_dim, seq_len=args.seq_len,
                 conv_dim=args.conv_dim, end_dim=args.end_dim,
                 adj_mx=adj_mx,
                 time_0=0.9, step_size_0=0.45,
                 time_1=0.9, step_size_1=0.9,
                 time_2=0.9, step_size_2=0.3,)
    blackbox.load_state_dict(torch.load(args.blackbox_file))
    
    engine = EngineXAI(scaler,
                       model,
                       blackbox,
                       perturbate_node,
                       degreeCentrality,
                       closenessCentrality,
                       pertubate,
                       args.num_nodes, 
                       args.learning_rate, args.weight_decay, 
                       device, 
                       adj_mx,
                       train_loss=args.train_loss,
                       test_loss=args.test_loss)
    
    # load checkpoints
    if args.epochs != 0 and args.iter_epoch != -1:
        print('loading epoch {}'.format(args.iter_epoch))
        model.load_state_dict(torch.load(args.save + '/G_T_model_{}.pth'.format(args.iter_epoch)))
    
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    
    log_file_train = open('loss_train_log.txt', 'w')
    log_file_val = open('loss_val_log.txt', 'w')
    log_file_test = open('loss_test_log.txt', 'w')
    
    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    best_epoch = 0 if args.epochs != 0 else args.iter_epoch

    # train loop
    for i in range(args.iter_epoch + 1, args.iter_epoch + 1 + args.epochs):
        train_loss = []
        train_mape = []
        train_rmse = []
        
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        

        for iter, (x, _) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)
            
            metrics = engine.train(trainx)
            
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: ' + '{:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)
        t2 = time.time()
        train_time.append(t2 - t1)
        
        valid_loss = []
        valid_mape = []
        valid_rmse = []
        s1 = time.time()
        for iter, (x, _) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            metrics = engine.eval(testx)
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        
        # test loop
        test_loss = []
        test_mape = []
        test_rmse = []
        for iter, (x, _) in enumerate(dataloader['test_loader'].get_iterator()):
            testx_ = torch.Tensor(x).to(device)
            testx_ = testx_.transpose(1, 3)
            metrics = engine.eval(testx_)
            test_loss.append(metrics[0])
            test_mape.append(metrics[1])
            test_rmse.append(metrics[2])
           
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        
        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        
        mtest_loss = np.mean(test_loss)
        mtest_mape = np.mean(test_mape)
        mtest_rmse = np.mean(test_rmse)
        
        his_loss.append(mvalid_loss)
        torch.save(engine.model.state_dict(), args.save + "/G_T_model_" + str(i) + ".pth")
        if np.argmin(his_loss) == len(his_loss) - 1:
            best_epoch = i
        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, ' + \
              'Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, ' + \
              'Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, 
                         mtrain_loss, 
                         mtrain_mape, 
                         mtrain_rmse, 
                         mvalid_loss,
                         mvalid_mape, 
                         mvalid_rmse, 
                         mtest_loss,
                         mtest_mape,
                         mtest_rmse,
                         (t2 - t1)))
        log_file_train.write(f'Epoch {i}, Training Loss: {mtrain_loss:.4f}, Training MAPE: {mtrain_mape:.4f}, Training RMSE: {mtrain_rmse:.4f} \n')
        log_file_train.flush()
        log_file_val.write(f'Epoch {i}, Val Loss: {mvalid_loss:.4f}, Val MAPE: {mvalid_mape:.4f}, Val RMSE: {mvalid_rmse:.4f} \n')
        log_file_val.flush()
        log_file_test.write(f'Epoch {i}, Test Loss: {mtest_loss:.4f}, Test MAPE: {mtest_mape:.4f}, Test RMSE: {mtest_rmse:.4f} \n')
        log_file_test.flush()
    
    if args.epochs != 0:
        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
        print("Training finished")
    else:
        print("skipped training")
    # horizon
    if args.epochs != 0:
        print('testing horizon on epoch (best epoch) {}'.format({best_epoch}))
        test_horizon(best_epoch, engine, device, adj_mx, dataloader, scaler, args)
    else:
        if args.single_test == True:
            print('testing horizon on epoch {}'.format({args.iter_epoch}))
            test_horizon(args.iter_epoch, engine, device, adj_mx, dataloader, scaler, args)
        else:
            log_horizon_test = open('horizon_log.txt', 'w')
            for epoch in range(5, args.iter_epoch + 1):
                print('testing horizon on epoch {}'.format({epoch}))
                log = test_horizon(epoch, engine, device, adj_mx, dataloader, scaler, args)
                log_horizon_test.write('epoch {}: '.format(epoch) + log + '\n')
                log_file_train.flush()

def test_horizon(epoch, engine, device, adj_mx, dataloader, scaler, args):
    # horizon test
    adj_mx = torch.tensor(adj_mx).to(device)
    engine.model.load_state_dict(torch.load(args.save + "/G_T_model_" + str(epoch) + ".pth"))
    outputs = []
    labels = []
    
    for iter, (x, _) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).transpose(1, 3).to(device)
        with torch.no_grad():
            Y, Ym = engine.horizon_forward(testx)
            Y = Y.transpose(1, 3)[:, 0, :, :]
            Ym = Ym.transpose(1, 3)[:, 0, :, :]
        outputs.append(Ym.squeeze())
        labels.append(Y.squeeze())
    
    # [n_samples, num_nodes, time_steps]
    realy = torch.cat(labels, dim=0)
    yhat = torch.cat(outputs, dim=0)
    
    amae = []
    amape = []
    armse = []
    for i in range(12):
        pred = yhat[:, :, i]
        real = realy[:, :, i]
        mae = util.masked_mae(pred, real)
        mape = util.masked_mape(scaler.inverse_transform(pred), scaler.inverse_transform(real))
        rmse = util.masked_rmse(pred, real)
        metrics = [mae.item(), mape.item(), rmse.item()]
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE:' + '{:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
    log = ('On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: ' + '{:.4f}, Test RMSE: {:.4f}').format(np.mean(amae), np.mean(amape), np.mean(armse))
    print(log)
    return log

if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))