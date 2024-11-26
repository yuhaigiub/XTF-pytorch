import argparse
import os
import time

import numpy as np
import torch
import util

from engine import Engine

parser = argparse.ArgumentParser()

"""
epoch == 0: test horizon only
    + single_test == True => <load iter_epoch> + test horizon
    + single_test == False => test horizon from epoch 0 to <iter_epoch>
epoch > 0: training
    + iter_epoch == -1 => start fresh (train from epoch 0)
    + iter_epoch > -1 => load <iter_epoch> and continue training
"""

MODEL_CHOICES = ['mnode', 
                 'graphwavenet', 
                 'mtgode', 
                 'stgode', 
                 'staeformer', 
                 'mtgnn', 
                 'stgcn', 
                 'gman', 
                 'mra_bgcn',
                 'dcrnn',
                 'stg_ncde',
                 'stid']
DATASET_CHOICES = ['METR-LA', 'PEMS-BAY', 'PEMS03', 'PEMS04', 'PEMS07', 'PEMS08']

# data settings
parser.add_argument('--device', type=str, default='cuda', help='device to run the model on')
parser.add_argument('--model_name', type=str, default='mnode',
                    choices=MODEL_CHOICES)
parser.add_argument('--dataset', type=str, default='PEMS08', 
                    choices=DATASET_CHOICES)

parser.add_argument('--blackbox_file', type=str, 
                    default='save_blackbox/G_T_model_1.pth', 
                    help='blackbox .pth file')

# training settings
parser.add_argument('--batch_size', type=int, default=32, help='settings')

parser.add_argument('--epochs', type=int, default=100, help='settings')
parser.add_argument('--iter_epoch', type=int, default=-1, help='using for save pth file')
parser.add_argument('--single_test', type=bool, default=True, help='settings')

parser.add_argument('--train_loss', type=str, default='huber', choices=['mae', 'huber'])
parser.add_argument('--test_loss', type=str, default='mae', choices=['mae', 'huber'])

parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--scheduler', type=bool, default=False)
# parser.add_argument('--early_stopping', type=int, default=30)

parser.add_argument('--print_every', type=int, default=50, help='')

# mnode settings
parser.add_argument('--n_cells', type=int, default=3, help='number of ensemble cells')
parser.add_argument('--conv_dim', type=int, default=32, help='settings')
parser.add_argument('--end_dim', type=int, default=64, help='settings')

#
parser.add_argument('--seq_len', type=int, default=12, help='time step')
parser.add_argument('--out_dim', type=int, default=1)
parser.add_argument('--huber_delta', type=float, default=1.0)
parser.add_argument('--setting_id', type=str, default="Int3")

parser.add_argument('--combine_method', type=str, default='mean', choices=['sum', 'mean'])
parser.add_argument('--decoder_method', type=str, default='1', choices=['1', '2'])

args = parser.parse_args()
args.data = './store/{}'.format(args.dataset)
args.adjdata = './store/{}/adj_mx.pkl'.format(args.dataset)
args.save = './saved_models/{}/{}'.format(args.dataset, args.model_name)

def main():
    print('dataset:', args.dataset)
    device = torch.device(args.device)
    
    # dataset args settings ------------------------------
    if args.dataset in DATASET_CHOICES[0:2]: 
        scaled_indices = (0, )
        args.out_feat = 0
        args.in_dim = 2
        # args.learning_rate = 0.001
        # args.weight_decay = 0.0001
        args.milestone = [20, 30]
    elif args.dataset in DATASET_CHOICES[2:]:
        scaled_indices = (0, )
        args.out_feat = 0
        args.in_dim = 3
        # args.learning_rate = 0.001
        # args.weight_decay = 0.0001
        args.milestone = [24, 45, 65]
    else:
        raise Exception("Invalid dataset")
    
    # load_adj ------------------------------
    if args.dataset in DATASET_CHOICES[0:2]:
        _, _, adj_mx = util.load_adj(args.adjdata)
    else:
        adj_mx = util.load_adj(args.adjdata)
    args.num_nodes = adj_mx.shape[0]
    
    if args.dataset in DATASET_CHOICES[0:2]:
        # remove self loop
        adj_mx = adj_mx - np.eye(args.num_nodes, dtype=np.float32)
    
    # if args.dataset in ['PEMS04', 'PEMS08']:
        # add self loop
        # adj_mx = adj_mx + np.eye(args.num_nodes, dtype=np.float32)
    
    # load dataset ------------------------------
    dataloader = util.load_dataset(args.data, 
                                   args.batch_size, 
                                   args.batch_size, 
                                   args.batch_size, 
                                   scaled_indices)
    
    # baseline-specific settings ------------------------------------------------------------
    if args.model_name == 'stgode':
        dtw_matrix, sp_matrix = util.stgode_adj_load(args.dataset)
        A_sp_wave = util.stgode_normalized_adj(sp_matrix).to(device)
        A_se_wave = util.stgode_normalized_adj(dtw_matrix).to(device)
    elif args.model_name == 'gman':
        SE, trainTE, valTE, testTE = util.gman_data_preprocess(dataloader, args.num_nodes)
        trainTE = trainTE.to(device)
        valTE = valTE.to(device)
        testTE = testTE.to(device)
        SE = SE.to(device)
        
        print(trainTE.shape, valTE.shape, testTE.shape)
    
    # Mean / std dev scaling is performed to the model output
    scaler = dataloader['scaler']
    
    
    # load model ------------------------------------------------------------
    if args.model_name == 'mnode':
        from alblation2.mste_Int4 import MSTE
        MODEL_PARAM = {
            "num_nodes": adj_mx.shape[0], 
            "seq_len": args.seq_len, 
            "in_dim": args.in_dim, 
            "conv_dim": 30, 
            "end_dim": 64,
            "adj_mx": adj_mx,
            "time_1": 0.9, 
            "step_size_1": 0.3,
            "time_2": 0.9, 
            "step_size_2": 0.3,
            "expert_steps": [
                [(0, 5, 11), (0, 3, 5, 7, 9, 11), tuple(range(12))],
                [(0, 6, 11), (0, 2, 4, 6, 8, 11), tuple(range(12))],
                [(0, 5, 11), (0, 3, 5, 7, 9, 11), tuple(range(12))],
                # [tuple(range(12)), tuple(range(12)), tuple(range(12))]
            ],
            "dropout": 0.3
        }
        model = MSTE(**MODEL_PARAM)
    elif args.model_name == 'graphwavenet':
        from baseline.graphwavenet.model import GraphWaveNet
        
        model = GraphWaveNet(num_nodes=args.num_nodes, 
                             in_channels=args.in_dim, 
                             out_channels=args.out_dim, 
                             out_timesteps=args.seq_len)
    elif args.model_name == 'stgode':
        from baseline.stgode.model import ODEGCN
        
        model = ODEGCN(num_nodes=args.num_nodes,
                       num_features=args.in_dim, 
                       num_timesteps_input=args.seq_len, 
                       num_timesteps_output=args.seq_len, 
                       A_sp_hat=A_sp_wave,
                       A_se_hat=A_se_wave)
    elif args.model_name == 'staeformer':
        from baseline.staeformer.model import STAEformer
        
        model = STAEformer(num_nodes=args.num_nodes, 
                           in_steps=args.seq_len,
                           out_steps=args.seq_len,
                           input_dim=args.in_dim,
                           output_dim=args.out_dim)
    elif args.model_name == 'mtgode':
        from baseline.mtgode.model import MTGODE
        
        model = MTGODE(device=device, 
                       in_dim=args.in_dim,
                       out_dim=args.out_dim,
                       predefined_A=torch.tensor(adj_mx).to(device), 
                       num_nodes=adj_mx.shape[0], 
                       buildA_true=False)
    elif args.model_name == 'gman':
        from baseline.gman.model import GMAN
        
        model = GMAN(in_dim=args.in_dim, out_dim=args.out_dim,
                     SE=SE, 
                     L=1, K=8, d=8, 
                     seq_len=args.seq_len, bn_decay=0.1)
    elif args.model_name == 'mtgnn':
        from baseline.mtgnn.model import MTGNN
        
        predefined_A = torch.tensor(adj_mx)
        # predefined_A = predefined_A - torch.eye(args.num_nodes)
        
        predefined_A = predefined_A.to(device)
        model = MTGNN(gcn_true=True,
                      buildA_true=False,
                      gcn_depth=1,
                      num_nodes=args.num_nodes,
                      device=device,
                      predefined_A=predefined_A,
                      node_dim=80,
                      in_dim=args.in_dim,
                      out_dim=args.out_dim,
                      seq_length=args.seq_len)
    elif args.model_name == 'stgcn':
        from baseline.stgcn.model import STGCNParams, STGCNChebGraphConv as STGCN
        
        params = STGCNParams(args.seq_len)
        gso = util.stgcn_calc_gso(adj_mx, params.gso_type)
        gso = util.stgcn_calc_chebynet_gso(gso)
        gso = gso.toarray().astype(dtype=np.float32)
        params.gso = torch.from_numpy(gso).to(device)
        
        Ko = params.n_his - (params.Kt - 1) * 2 * params.stblock_num
        blocks = []
        blocks.append([args.in_dim])
        for l in range(params.stblock_num):
            blocks.append([64, 16, 64])
        if Ko == 0:
            blocks.append([128])
        elif Ko > 0:
            blocks.append([128, 128])
        blocks.append([args.seq_len])
        print('blocks:', blocks)
        model = STGCN(params, blocks, args.num_nodes)
    elif args.model_name == 'dcrnn':
        from baseline.dcrnn.model import DCRNN
        
        params = {
            "filter_type": "dual_random_walk",
            "cl_decay_steps": 2000,
            "horizon": args.seq_len,
            "seq_len": args.seq_len,
            "input_dim": args.in_dim,
            "output_dim": args.out_dim,
            "l1_decay": 0,
            "max_diffusion_step": 2,
            "num_nodes": args.num_nodes,
            "num_rnn_layers": 2,
            "rnn_units": 64,
            "use_curriculum_learning": True,
            "device": device
        }
        model = DCRNN(adj_mx, **params)
    elif args.model_name == 'mra_bgcn':
        from baseline.mra_bgcn.model import MRA_BGCN_Encoder_Decoder
        import networkx as nx
        
        adj_node = adj_mx
        
        G = nx.from_numpy_array(adj_mx)
        L = nx.line_graph(G)
        adj_edge = nx.adjacency_matrix(L).toarray()
        # edge_lim = {
        #     'METR-LA': (1520, 500),
        #     'PEMS-BAY': (2404, 500),
        #     'PEMS04': (209, 209),
        #     'PEMS08': (135, 135),
        # }
        
        M = torch.tensor(nx.incidence_matrix(G).toarray(), dtype=torch.float32).to(device)
        model = MRA_BGCN_Encoder_Decoder(args.in_dim, 
                                         args.out_dim,
                                         args.seq_len,
                                         64, 32,
                                         adj_node, adj_edge,
                                         64, 32,
                                         M=M, range_K=3,
                                         residual=True,
                                         device=device)
    elif args.model_name == 'stg_ncde':
        from baseline.stg_ncde import vector_fields
        from baseline.stg_ncde.model import NeuralGCDE, NeuralGCDEParams
        
        hidden_dim = 32
        hidden_dim_2 = 64
        embed_dim = 10
        num_layer=2
        
        params = NeuralGCDEParams(args.num_nodes, args.seq_len, num_layer, True, embed_dim)
        
        vector_fields_f = vector_fields.FinalTanh_f(args.in_dim, hidden_dim, hidden_dim_2,num_hidden_layers=2)
        vector_fields_g = vector_fields.VectorField_g(args.in_dim, hidden_dim, hidden_dim_2, 
                                                      num_hidden_layers=num_layer,
                                                      num_nodes=args.num_nodes,
                                                      cheb_k=2,
                                                      embed_dim=embed_dim,
                                                      g_type='agc')
        
        model = NeuralGCDE(args=params, 
                           func_f=vector_fields_f, 
                           func_g=vector_fields_g, 
                           input_channels=args.in_dim,
                           hidden_channels=32,
                           output_channels=args.out_dim,
                           initial=True,
                           device=device,
                           atol=1e-9, rtol=1e-7, solver="rk4")
    elif args.model_name == 'stid':
        from baseline.stid.model import STID
        
        args.in_dim = 3
        
        params = {
            "num_nodes": args.num_nodes,
            "input_len": args.seq_len,
            "input_dim": args.in_dim,
            "embed_dim": 128,
            "output_len": args.seq_len,
            "num_layer": 3,
            "if_node": True,
            "node_dim": 32,
            "if_T_i_D": True,
            "if_D_i_W": True,
            "temp_dim_tid": 32,
            "temp_dim_diw": 32,
            "time_of_day_size": 288,
            "day_of_week_size": 7
        }
        
        model = STID(**params)
    else:
        raise Exception("Invalid model")
    
    # calculate model size (in MB) ------------------------------
    if args.model_name != "dcrnn":
        model_size = util.calculate_model_size(model)
        print('model size: {:.3f}MB'.format(model_size))
    
    # load Engine ------------------------------------------------------------
    engine = Engine(scaler,
                    model,
                    args.num_nodes, 
                    args.learning_rate,
                    args.weight_decay, 
                    args.milestone,
                    device, 
                    adj_mx,
                    args.out_feat,
                    args.model_name,
                    train_loss=args.train_loss,
                    test_loss=args.test_loss,
                    huber_delta=args.huber_delta,
                    scheduleLR=args.scheduler)
    
    # load checkpoints ------------------------------------------------------------
    if args.epochs != 0 and args.iter_epoch != -1:
        print('loading epoch {}'.format(args.iter_epoch))
        model.load_state_dict(torch.load(args.save + '/G_T_model_{}.pth'.format(args.iter_epoch)))
    
    
    # prepare log files ------------------------------------------------------------
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    
    log_file_train = open('loss_train_log.txt', 'w')
    log_file_val = open('loss_val_log.txt', 'w')
    log_file_test = open('loss_test_log.txt', 'w')
    
    # ------------------------------------------------------------------------------------------
    
    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    best_epoch = 0 if args.epochs != 0 else args.iter_epoch
    
    if args.model_name == 'dcrnn':
        batch_seen = dataloader['train_loader'].num_batch * (args.iter_epoch + 1)
    
    # train loop ------------------------------------------------------------
    for i in range(args.iter_epoch + 1, args.iter_epoch + 1 + args.epochs):
        train_loss = []
        train_mape = []
        train_rmse = []
        
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainy = torch.Tensor(y).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = trainy.transpose(1, 3)
            
            kwargs = {}
            if args.model_name == "gman":
                start_idx = iter * args.batch_size
                end_idx = min(dataloader['x_train'].shape[0], (iter + 1) * args.batch_size)
                TE = trainTE[start_idx:end_idx]
                kwargs['gman_TE'] = TE
                
                trainx = trainx[:TE.shape[0], ...]
                trainy = trainy[:TE.shape[0], ...]
            elif args.model_name == "dcrnn":
                kwargs['batch_seen'] = batch_seen
            
            metrics = engine.train(trainx, trainy[:, args.out_feat, :, :], **kwargs)
            
            if args.model_name == "dcrnn":
                batch_seen += 1
            
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: ' + '{:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)
        
        t2 = time.time()
        train_time.append(t2 - t1)
        print(f'finish train loop. Time: {t2-t1:.2f}')
        # validate loop ------------------------------------------------------------
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testy = torch.Tensor(y).to(device)
            testx = testx.transpose(1, 3)
            testy = testy.transpose(1, 3)
            
            kwargs = {}
            if args.model_name == "gman":
                start_idx = iter * args.batch_size
                end_idx = min(dataloader['x_val'].shape[0], (iter + 1) * args.batch_size)
                TE = valTE[start_idx:end_idx]
                kwargs['gman_TE'] = TE
                
                testx = testx[:TE.shape[0], ...]
                testy = testy[:TE.shape[0], ...]
            
            metrics = engine.eval(testx, testy[:, args.out_feat, :, :], **kwargs)
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        
        # test loop ------------------------------------------------------------
        test_loss = []
        test_mape = []
        test_rmse = []
        
        for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx_ = torch.Tensor(x).to(device)
            testy_ = torch.Tensor(y).to(device)
            testx_ = testx_.transpose(1, 3)
            testy_ = testy_.transpose(1, 3)
            
            kwargs = {}
            if args.model_name == 'gman':
                start_idx = iter * args.batch_size
                end_idx = min(dataloader['x_test'].shape[0], (iter + 1) * args.batch_size)
                TE = testTE[start_idx:end_idx]
                kwargs['gman_TE'] = TE
                testx_ = testx_[:TE.shape[0], ...]
                testy_ = testy_[:TE.shape[0], ...]
            
            metrics = engine.eval(testx_, testy_[:, args.out_feat, :, :], **kwargs)
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
        
        
        if np.argmin(his_loss) == len(his_loss) - 1:
            # only save good epochs
            torch.save(engine.model.state_dict(), args.save + "/G_T_model_" + str(i) + ".pth")
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
    
    # horizon ------------------------------------------------------------
    
    if args.model_name == 'gman':
        args.testTE = testTE
    
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

def test_horizon(epoch, engine: Engine, device, adj_mx, dataloader, scaler, args):
    # horizon test
    adj_mx = torch.tensor(adj_mx).to(device)
    engine.model.load_state_dict(torch.load(args.save + "/G_T_model_" + str(epoch) + ".pth"))
    outputs = []
    
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, args.out_feat, :, :]
    
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).transpose(1, 3).to(device)
        with torch.no_grad():
            kwargs = {}
            if args.model_name == 'gman':
                start_idx = iter * args.batch_size
                end_idx = min(dataloader['x_test'].shape[0], (iter + 1) * args.batch_size)
                TE = args.testTE[start_idx:end_idx]
                kwargs['gman_TE'] = TE
                testx = testx[:TE.shape[0], ...]
            
            out = engine.simple_forward(testx, is_train=False, **kwargs) # [b, t, n, c]
            out = out.transpose(1, 3)[:, 0, :, :]
        outputs.append(out.squeeze())
    
    yhat = torch.cat(outputs, dim=0)
    
    if args.model_name == 'gman':
        min_size = min(yhat.size(0), realy.size(0))
        yhat = yhat[:min_size, ...]
        realy = realy[:min_size, ...]
    else:
        yhat = yhat[:realy.size(0), ...]
    
    amae = []
    amape = []
    armse = []
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:, :, i], i=engine.out_feat)
        real = realy[:, :, i]
        
        metrics = util.metric(pred, real)
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