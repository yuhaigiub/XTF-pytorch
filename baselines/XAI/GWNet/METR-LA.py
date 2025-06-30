from .arch import GraphWaveNet
from basicts.utils import get_regular_settings, load_adj
from basicts.scaler import ZScoreScaler
from .runners.runner_zoo.simple_tsf_runner import SimpleTimeSeriesForecastingRunner
from basicts.data import TimeSeriesForecastingDataset
from basicts.metrics import masked_mae, masked_mape, masked_rmse
import os
import sys
import torch
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../..'))


############################## Hot Parameters ##############################
# Dataset & Metrics configuration
DATA_NAME = 'METR-LA'  # Dataset name
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']  # Length of input sequence
OUTPUT_LEN = regular_settings['OUTPUT_LEN']  # Length of output sequence
# Train/Validation/Test split ratios
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
# Whether to normalize each channel of the data
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']
RESCALE = regular_settings['RESCALE']  # Whether to rescale the data
NULL_VAL = regular_settings['NULL_VAL']  # Null value in the data
# Model architecture and parameters
MODEL_ARCH = GraphWaveNet
adj_mx, _ = load_adj("datasets/" + DATA_NAME +
                     "/adj_mx.pkl", "doubletransition")
MODEL_PARAM = {
    "num_nodes": 207,
    "supports": [torch.tensor(i) for i in adj_mx],
    "dropout": 0.3,
    "gcn_bool": True,
    "addaptadj": True,
    "aptinit": None,
    "in_dim": 2,
    "out_dim": 12,
    "residual_channels": 32,
    "dilation_channels": 32,
    "skip_channels": 256,
    "end_channels": 512,
    "kernel_size": 2,
    "blocks": 4,
    "layers": 2
}
model = GraphWaveNet(num_nodes=MODEL_PARAM['num_nodes'], 
                     supports= MODEL_PARAM['supports'],
                     dropout=MODEL_PARAM['dropout'], 
                     gcn_bool=MODEL_PARAM['gcn_bool'],
                     addaptadj=MODEL_PARAM['addaptadj'],
                     aptinit=MODEL_PARAM['aptinit'],
                     in_dim=MODEL_PARAM['in_dim'],
                     out_dim=MODEL_PARAM['out_dim'],
                     residual_channels=MODEL_PARAM['residual_channels'],
                     dilation_channels=MODEL_PARAM['dilation_channels'],
                     skip_channels=MODEL_PARAM['skip_channels'],
                     end_channels=MODEL_PARAM['end_channels'],
                     kernel_size=MODEL_PARAM['kernel_size'],
                     blocks=MODEL_PARAM['blocks'],
                     layers=MODEL_PARAM['layers'])

NUM_EPOCHS = 1
# checkpoint = torch.load(r'D:\X_XFT\XTF-pytorch-v2\baselines\XAI\test.pt')
# if 'model_state_dict' in checkpoint:
#     model.load_state_dict(checkpoint['model_state_dict'])
#     print("Model weights loaded successfully.")
# else:
#     print("Model weights not found in the checkpoint.")
############################## General Configuration ##############################
CFG = EasyDict()
# General settings
CFG.DESCRIPTION = 'An Example Config'
# Number of GPUs to use (0 for CPU mode)
# CFG.GPU_NUM = torch.cuda.device_count()
CFG.GPU_NUM = 1

# Runner
CFG.RUNNER = SimpleTimeSeriesForecastingRunner
 
############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
# Dataset settings
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,
    # 'mode' is automatically set by the runner
})

############################## Scaler Configuration ##############################
CFG.SCALER = EasyDict()
# Scaler settings
CFG.SCALER.TYPE = ZScoreScaler  # Scaler class
CFG.SCALER.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_ratio': TRAIN_VAL_TEST_RATIO[0],
    'norm_each_channel': NORM_EACH_CHANNEL,
    'rescale': RESCALE,
})

############################## Model Configuration ##############################
CFG.MODEL = EasyDict()
# Model settings
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 1]
CFG.MODEL.TARGET_FEATURES = [0]
CFG.MODEL.DDP_FIND_UNUSED_PARAMETERS = True

#Blachbox setting
CFG.BLACKBOX = EasyDict()
CFG.BLACKBOX.NAME = MODEL_ARCH.__name__
CFG.BLACKBOX.ARCH = MODEL_ARCH
CFG.BLACKBOX.PARAM = MODEL_PARAM
CFG.BLACKBOX.FORWARD_FEATURES = [0, 1]
CFG.BLACKBOX.TARGET_FEATURES = [0]
CFG.BLACKBOX.DDP_FIND_UNUSED_PARAMETERS = True

############################## Metrics Configuration ##############################

CFG.METRICS = EasyDict()
# Metrics settings
CFG.METRICS.FUNCS = EasyDict({
    'MAE': masked_mae,
    'MAPE': masked_mape,
    'RMSE': masked_rmse,
})
CFG.METRICS.TARGET = 'MAE'
CFG.METRICS.NULL_VAL = NULL_VAL

############################## Training Configuration ##############################
CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    MODEL_ARCH.__name__,
    '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS),
             str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = masked_mae
# Optimizer settings
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.002,
    "weight_decay": 0.0001,
}
# Learning rate scheduler settings
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [1, 50],
    "gamma": 0.5
}
# Train data loader settings
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 16
CFG.TRAIN.DATA.SHUFFLE = True
# Gradient clipping settings
CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 5.0
}

############################## Validation Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 16

############################## Test Configuration ##############################
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 16

############################## Evaluation Configuration ##############################

CFG.EVAL = EasyDict()

# Evaluation parameters
# Prediction horizons for evaluation. Default: []
CFG.EVAL.HORIZONS = [3, 6, 12]
CFG.EVAL.USE_GPU = True  # Whether to use GPU for evaluation. Default: True

