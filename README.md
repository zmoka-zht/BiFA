# BiFA: Remote Sensing Image Change Detection with Bitemporal Feature Alignment

This is the pytorch implement of our paper "BiFA: Remote Sensing Image Change Detection with Bitemporal Feature Alignment"


[Project Page](https://github.com/zmoka-zht/BiFA/tree/main/) $\cdot$ [PDF Download]() $\cdot$ [HuggingFace Demo]()


## 0. Environment Setup

### 0.1 Create a virtual environment

```shell
conda create -n BiFA python=3.7
```

### 0.2 Activate the virtual environment
```sehll
conda activate BiFA
```

### 0.3  Install pytorch
```shell
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
```

### 0.4 Install mmcv
```shell
pip install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html
```
Please refer to [installation documentation](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) for more detailed installation.

### 0.5 Install other dependencies
```shell
pip install -r requirements.txt
```

## 1. Data Preparation

### 1.1 Data structure

```
"""
Change detection data set with pixel-level binary labels；
├─A
├─B
├─label
└─list
"""
```
`A`: images of t1 phase;

`B`:images of t2 phase;

`label`: label maps;

`list`: contains `train.txt, val.txt and test.txt`, each file records the image names (XXX.png) in the change detection dataset.

### 1.2 Data download
LEVIR-CD: https://justchenhao.github.io/LEVIR/

WHU-CD: https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html

DSIFN-CD: https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images/tree/master/dataset


## 2. Model Training

### 1.1 BiFA
#### 1.1.1 Config file
The config file of BiFA is **config/xxx_bifa.json**. You can modify the parameters in this file according to the situation.

#### 1.1.2 Training
Run ```python train_cd.py``` to train the BiFA model. And you can modify the ArgumentParser parameters in this file according to the situation.

## 3. Model Evaluation

### 3.1 BiFA
#### 3.1.1 Config file
The config file of FunSR is **config/xxx_test_bifa.json**. You can modify the parameters in this file according to the situation.
#### 3.1.2 Testing
Run ```python test_cd.py``` to test the FunSR model. And you can modify the ArgumentParser parameters in this file according to the situation.


## 4. Model Download
The model weights of BiFA are provided in the **experiments/pretrain**

## 5. Citation
If you find this project useful for your research, please cite our paper.

If you have any other questions, please contact me!!!
