# Cyclic Contrastive Knowledge Transfer for Open-Vocabulary Object Detection

*2024*

[GitHub Repository](https://anonymous.4open.science/r/CCKT-Det-C7B7) | 

## Overview

We propose **CCKT-Det**. 

## Installation
Our models are set under `python=3.9`, `pytorch=2.4.1` . Other versions might be available as well.

1. Compiling CUDA operators as [deformable-detr](https://github.com/fundamentalvision/Deformable-DETR) 
2. Install other packages including [open-clip](https://github.com/mlfoundations/open_clip), [coco-api](https://github.com/cocodataset/cocoapi),  `mmdet`,  `timm`,  `mmcv-full`

## Data
For OVD-COCO setting, Please follow download [COCO2017](https://cocodataset.org/#home) dataset and follow [OVR-CNN](https://github.com/alirezazareian/ovr-cnn) to split data into base and novel class.

The data file is organised as following:

```
coco_path/
├── train2017/
├── val2017/
└── annotations/
├── instances_train2017_base.json
└── instances_val2017_all.json
```

## Run

#### To train a model using 8 cards

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --with_box_refine --output_dir outputs/
```



#### To evaluate a model using 8 cards

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --with_box_refine --output_dir outputs/ --eval --resume outputs/checkpoint.pth
```

