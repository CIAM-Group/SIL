

## [ICLR25] Boosting Neural Combinatorial Optimization for Large-Scale Vehicle Routing Problems

This repository contains the code implementation of paper [Boosting Neural Combinatorial Optimization for Large-Scale Vehicle Routing Problems](https://openreview.net/forum?id=TbTJJNjumY). In this paper, we propose a lightweight cross-attention mechanism with linear complexity, by which a Transformer network is developed to learn efficient and favorable solutions for large-scale VRPs. We also propose a Self-Improved Training (SIT) algorithm that enables direct model training on large-scale VRP instances, bypassing extensive computational overhead for attaining labels. 

### Dependencies
```bash
Python=3.8.6
matplotlib==3.5.2
numpy==1.23.3
pandas==1.5.1
pytz==2022.1
torch==1.12.1
torchaudio==0.12.1
torchvision==0.13.1
tqdm==4.64.1
```
If any package is missing, just install it following the prompts.

### Datasets and pre-trained models
The training and test datasets can be downloaded from Google Drive:
```bash
https://drive.google.com/drive/folders/1SEk34Sws_cwE9PYAkdHZeYXZN_b6o-db?usp=sharing
```
or  Baidu Cloud:
```bash
https://pan.baidu.com/s/1NzLtoPl-1i77_JSinSKoXw?pwd=zbez
```

### Implementation

#### Testing
Run `Test_All\test.py` and `Test_All\test_PRC.py` to test the model using `greedy search` and `parallel random reconstruction`, respectively.

#### Training

To train the model, run `Train\Train.train.py`. This file contains parameters you can modify. Some instructions on the training process are provided.


## Citation

**If this repository is helpful for your research, please cite our paper:<br />**
*"Fu Luo, Xi Lin, Yaoxin Wu, Zhenkun Wang, Tong Xialiang, Mingxuan Yuan, and Qingfu Zhang, Boosting Neural Combinatorial Optimization for Large-Scale Vehicle Routing Problems, The Thirteenth International Conference on Learning Representations, ICLR 2025" <br />*

**OR**

```
@inproceedings{
luo2025boosting,
title={Boosting Neural Combinatorial Optimization for Large-Scale Vehicle Routing Problems},
author={Fu Luo and Xi Lin and Yaoxin Wu and Zhenkun Wang and Tong Xialiang and Mingxuan Yuan and Qingfu Zhang},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=TbTJJNjumY}
}
```
****


## Acknowledgements
This work extends our previous research presented in [LEHD](https://github.com/CIAM-Group/NCO_code/tree/main/single_objective/LEHD). 
The implementation framework is adapted from [POMO](https://github.com/yd-kwon/POMO), with the random insertion module inherited from [GLOP](https://github.com/henry-yeh/GLOP) 's implementation. We acknowledge the foundational contributions of these open-source projects.
