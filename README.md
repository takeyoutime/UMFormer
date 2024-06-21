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
 
- Folder structure
  - Place the processed data set in the following order:
```none
Dataset
├── uavid
│   ├── uavid_train (original)
│   ├── uavid_val (original)
│   ├── uavid_test (original)
│   ├── uavid_train_val (Merge uavid_train and uavid_val)
│   ├── train (processed)
│   ├── val (processed)
│   ├── train_val (processed)
├── vaihingen
│   ├── train_images (original)
│   ├── train_masks (original)
│   ├── test_images (original)
│   ├── test_masks (original)
│   ├── test_masks_eroded (original)
│   ├── train (processed)
│   ├── test (processed)
├── potsdam (the same with vaihingen)
```


## Overview framework of UMFormer

We aim to design a precise semantic segmentation network for remote sensing images. Inspired by the self-attention mechanism and Mamba, we propose UMFormer, a network that fuses these two techniques within an encoder-decoder framework to address the aforementioned challenges. UMFormer combines the respective advantages of CNN, self-attention mechanism and Mamba to create a hybrid network effectively.

![Overview Framework of UMFormer](/Image/UMFormer.jpg)


## Comparison of network parameters


## Inference on huge remote sensing image
```
python GeoSeg/inference_huge_image.py \
-i data/vaihingen/test_images \
-c GeoSeg/config/vaihingen/dcswin.py \
-o fig_results/vaihingen/dcswin_huge \
-t 'lr' -ph 512 -pw 512 -b 2 -d "pv"
```

<div>
<img src="vai.png" width="30%"/>
<img src="pot.png" width="35.5%"/>
</div>

## Reproduction Results
|    Method     |  Dataset  |  F1   |  OA   |  mIoU |
|:-------------:|:---------:|:-----:|:-----:|------:|
|  UNetFormer   |   UAVid   |   -   |   -   | 67.63 |
|  UNetFormer   | Vaihingen | 90.30 | 91.10 | 82.54 |
|  UNetFormer   |  Potsdam  | 92.64 | 91.19 | 86.52 |
|  UNetFormer   |  LoveDA   |   -   |   -   | 52.97 |
| FT-UNetFormer | Vaihingen | 91.17 | 91.74 | 83.98 |
| FT-UNetFormer |  Potsdam  | 93.22 | 91.87 | 87.50 |

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
