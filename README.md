# XTF: eXplainable Traffic Forecasting Framework using Multiple Spatio-Temporal ODE Experts

## MSTE

Train
`python experiments/train.py -c baselines/MSTE/METR-LA.py -g 0,1`

Note: We set the default parameters `CFG.GPU_NUM = torch.cuda.device_count()` in `/baselines/MSTE/[dataset].py`. You can set this to be any value you want and adjust the `-g` flag accordingly (0 if you want to use CPU)

Evaluate
`python experiments/evaluate.py -cfg baselines/MSTE/METR-LA.py -ckpt pretrain/METR-LA/MSTE_best_val_CUSTOM.pt -g 0`

## XAI
