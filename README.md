## A lightweight semantic segmentation network based on self-attention mechanism and state space model for real-time urban scene segmentation

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


## Reproduction Results
|   Method   |  Dataset  |  F1  |  OA  | mIoU |
|:----------:|:---------:|:----:|:----:|-----:|
|  UMFormer  |   UAVid   | 79.3 | 85.7 | 67.7 |
|  UMFormer  | Vaihingen | 90.7 | 93.0 | 83.3 |
|  UMFormer  |  Potsdam  | 92.0 | 90.9 | 85.5 |

Due to some random operations in the training stage, reproduced results (run once) are slightly different from the reported in paper.

## Citation

If you find this project useful in your research, please consider citing：

- [UNetFormer: A UNet-like transformer for efficient semantic segmentation of remote sensing urban scene imagery](https://authors.elsevier.com/a/1fIji3I9x1j9Fs)
- [A Novel Transformer Based Semantic Segmentation Scheme for Fine-Resolution Remote Sensing Images](https://ieeexplore.ieee.org/abstract/document/9681903) 
- [Transformer Meets Convolution: A Bilateral Awareness Network for Semantic Segmentation of Very Fine Resolution Urban Scene Images](https://www.mdpi.com/2072-4292/13/16/3065)
- [ABCNet: Attentive Bilateral Contextual Network for Efficient Semantic Segmentation of Fine-Resolution Remote Sensing Images](https://www.sciencedirect.com/science/article/pii/S0924271621002379)
- [Multiattention network for semantic segmentation of fine-resolution remote sensing images](https://ieeexplore.ieee.org/abstract/document/9487010)
- [A2-FPN for semantic segmentation of fine-resolution remotely sensed images](https://www.tandfonline.com/doi/full/10.1080/01431161.2022.2030071)



## Acknowledgement

We wish **GeoSeg** could serve the growing research of remote sensing by providing a unified benchmark 
and inspiring researchers to develop their own segmentation networks. Many thanks the following projects's contributions to **GeoSeg**.
- [pytorch lightning](https://www.pytorchlightning.ai/)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)
- [ttach](https://github.com/qubvel/ttach)
- [catalyst](https://github.com/catalyst-team/catalyst)
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
