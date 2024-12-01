# XTF: eXplainable Traffic Forecasting Framework using Multiple Spatio-Temporal ODE Experts

### Abstract
Advances in technology have facilitated the trans formation of transportation systems through the deployment of sensor-equipped devices, generating extensive traffic data. Although recent traffic forecasting models, notably neural network based models, exhibit promising performance, they generally struggle to capture continuous spatio-temporal dynamics, are prone to overfitting, and lack transparency in their forecasting. To address these limitations of current methods, we introduce the eXplainable Traffic Forecasting framework (XTF). XTF employs hierarchical ordinary differential equations (ODEs) to continuously model spatio-temporal dynamics, facilitating more accurate representations of complex traffic patterns. To mitigate overfitting, XTF utilizes an ensemble approach that integrates bagging and stacking strategies, thereby improving generalization across diverse datasets. Finally, XTF includes an explanation plug-in that generates saliency maps, enhancing the transparency and interpretability of forecasting results. Empirical studies using four real-world datasets provide evidence that XTF is capable of substantial improvements over state-of-the-art methods in terms of both forecasting accuracy and interpretability.

###  Framework Overview
![ Framework Overview](figures/Framework_Overview.jpg)

### Requirements
To install requirements, run:
```bash
pip install -r requirements.txt
```

### Data Preparation
Download data from this repository [github](https://github.com/Jimmy-7664/STD-MAE) (The data file should be renamed to `data.h5` or `data.npz`). The data folder `store` should be organized as follow:
```bash
/store
|----/[Dataset Name]
|    |
|    |----adj_mx.pkl
|----/raw
     |
     |----/[Dataset Name]
          |
          |----data.[h5 or npz]
```
Then run this script to preprocess data:
```bash
python generate_data.py --dataset METR-LA
```

### Run MSTE
```bash
python run.py
```

### Run STEP
```bash
cd STEP
python main_XAI.p
