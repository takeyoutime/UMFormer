# A lightweight semantic segmentation network based on self-attention mechanism and state space model for real-time urban scene segmentation

## Introduction

Some samples with challenges in semantic segmentation of remote sensing images:

![Introduction](/Image/introduction.jpg)

## Dataset

- Supported Remote Sensing Datasets
  - [ISPRS Vaihingen and Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx) 
  - [UAVid](https://uavid.nl/)
  - More datasets will be supported in the future.
  
- Dataset preprocessing
  - Please refer to [**Geoseg**](https://github.com/WangLibo1995/GeoSeg)

## Overview framework of UMFormer

We aim to design a precise semantic segmentation network for remote sensing images. Inspired by the self-attention mechanism and Mamba, we propose UMFormer, a network that fuses these two techniques within an encoder-decoder framework to address the aforementioned challenges. UMFormer combines the respective advantages of CNN, self-attention mechanism and Mamba to create a hybrid network effectively.

![Overview Framework of UMFormer](/Image/UMFormer.jpg)

## Comparison of network parameters

In the context of remote sensing image semantic segmentation, the parameter size, complexity and speed of the network are also crucial evaluation indicators. In response, a comparison is made between UMFormer and efficient semantic segmentation networks, including the number of model parameters (M), the floating point operation count (FLOPs) and the frames per second (FPS).

![Comparison of network parameters](/Image/Comparison-of-network-parameters.jpg)

## Numerical results
|   Method   |  Dataset  |  F1  |  OA  | mIoU |
|:----------:|:---------:|:----:|:----:|-----:|
|  UMFormer  |   UAVid   | 79.3 | 85.7 | 67.7 |
|  UMFormer  | Vaihingen | 90.7 | 93.0 | 83.3 |
|  UMFormer  |  Potsdam  | 92.0 | 90.9 | 85.5 |


## Visualization

### UAVid
Visualization of the UAVid validation set. The first column represents the original images. The second column represents the ground truth. The third column represents the UNetFormer segmentation results. The fourth column represents the segmentation results of our method.
![UAVid](/Image/uavid.jpg)

### Vaihingen
Visualization of the Vaihingen validation set. The columns from left to right are: original images, ground truth, segmentation results of ABCNet, segmentation results of UNetFormer, segmentation results of Mamba-UNet, segmentation results of ours.
![UAVid](/Image/vaihingen.jpg)

### Potsdam
Visualization of the Potsdam validation set. The columns from left to right are: original images, ground truth, segmentation results of MAResU-Net, segmentation results of UNetFormer, segmentation results of Swin-UMamba, segmentation results of ours.
![UAVid](/Image/potsdam.jpg)


## Installation

Step 0: Create a conda environment.
```
conda create -n UMFormer python=3.8
conda activate UMFormer
```

Step 1: Install pytorch and torchvision matching your CUDA version:
```
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

Step 2: Install requirements:
```
pip install -r requirements.txt
```

Step 3: Install Mamba:
```
pip install mamba-ssm==1.2.0.post1

pip install causal-conv1d==1.2.0.post2
```

## Acknowledgement

Our code is based on the following previous work：  

[UNetformer](https://github.com/WangLibo1995/GeoSeg)  
[Visio Mamba](https://github.com/hustvl/Vim)  
[VM-UNet](https://github.com/JCruan519/VM-UNet/tree/main)

