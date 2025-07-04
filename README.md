# XTF: eXplainable Traffic Forecasting Framework using Multiple Spatio-Temporal ODE Experts

## Requirements
This project use python 3.10 + pytorch CUDA 12.6 or CUDA 12.8

After installing python and pytorch, run the following command to install all dependencies:

```bash
pip install -r requirements.txt
```

## MSTE

Train

```bash
python experiments/train.py -c baselines/MSTE/METR-LA.py -g 0,1
```

Note: We set the default parameters `CFG.GPU_NUM = torch.cuda.device_count()` in `/baselines/MSTE/[dataset].py`. You can set this to be any value you want (0 if you want to use CPU) and adjust the `-g` flag accordingly (Ex., `-g 0` to use GPU 0, `-g 0,1` to use GPU 0 and 1)

Evaluate

```bash
python experiments/evaluate.py -cfg baselines/MSTE/METR-LA.py -ckpt pretrain/METR-LA/MSTE_best_val_CUSTOM.pt -g 0
```

## XAI
Train

```bash
python experiments/train.py -c baselines/GWNET/METR-LA.py -g 0
```
