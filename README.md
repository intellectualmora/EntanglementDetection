
# Direct entanglement detection of quantum systems using machine learning
[![python](https://img.shields.io/badge/Python-3.8-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-1.13.1-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com) 
[![License: GPLv3](https://img.shields.io/badge/license-MIT-blue)](./LICENSE.txt)
[![arXiv](https://img.shields.io/badge/arXiv-2209.08501-b31b1b.svg)](https://arxiv.org/abs/2209.08501)
[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/chemora/EntanglementDetectionModel/tree/main)
[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-sm.svg)](https://huggingface.co/datasets/chemora/EntanglementDetectionDataSet/tree/main)
### Yulei Huang, Liangyu Che, Chao Wei, Feng Xu, Xinfang Nie, Jun Li, Dawei Lu and Tao Xin

In our recent [paper](https://arxiv.org/abs/2209.08501), we propose Direct entanglement detection of quantum systems using machine learning.

We provide the [pretrained models]() and [datasets](https://huggingface.co/datasets/chemora/EntanglementDetectionDataSet/tree/main).

## Pre-requisites
0. Python >= 3.8
0. Clone this repository
0. Install python requirements. Please refer [requirements.txt](requirements.txt)
0. Download datasets or use simulation code to generate datasets.
0. Configure training or inference parameters in `configs/config.json`
0. Train or download pretrained models.
0. Infer the test data.

## Installation
`conda` virtual environment is recommended. 
```
conda create -n EntanglementDetection python=3.8
conda activate EntanglementDetection
pip install -r requirements.txt
```

## Training Exmaple
```sh
python train.py

```


## Inference Example


## Acknowledgement


## Citation

If our code or models help your work, please cite our paper:
```BibTeX

```