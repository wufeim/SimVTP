

<div align="center">

# 🚀 **SimVTP**: Simple Video Text Pre-training with Masked Autoencoders

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a> [![License](https://img.shields.io/badge/arXiv-2212.03490-b31b1b.svg)](https://arxiv.org/abs/2212.03490) [![Python](https://img.shields.io/badge/Python-3.7-brightgreen.svg)](#PyTorch)

</div>

## Notes by Wufei

1. Install environment.

```sh
conda create -n simvtp pytorch=1.12.1 cudatoolkit=11.6 torchvision torchaudio jupyterlab --strict-channel-priority --override-channels -c https://aws-pytorch.s3.us-west-2.amazonaws.com -c pytorch -c nvidia -c conda-forge
conda activate simvtp
pip install timm==0.4.12 decord opencv-python tensorboardX einops transformers gdown
```

2. Download VideoMAE pretrained weights and modify the path in `run_simvtp_pretraining.py:L184`.

```sh
gdown https://drive.google.com/file/d/1tEhLyskjb755TJ65ptsrafUG2llSwQE1/view?usp=sharing --fuzzy
mkdir videomae_weights && mv checkpoint.pth videomae_weights/k400_vitb_1600_pretrain.pth
```

3. Download `vocab.txt` and modify the path to `vocab.txt` in `run_simvtp_vis.py:L159`.

```sh
wget https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt
mkdir bert-base-uncased-vocab && mv vocab.txt bert-base-uncased-vocab/vocab.txt
```

## 🍃 Abstract
SimVTP: a Simple Video-Text Pretraining framework via masked autoencoders. We randomly mask out the spatial-temporal tubes of input video and the word tokens of input text and then feed them into a unified autencoder to reconstruct the missing pixels and words. 

Our SimVTP has several properties: 
- Thanks to the unified autoencoder,  SimVTP reconstructs the masked signal of one modality with the help from another modality, which implicitly learns the cross-modal alignment between video tubes and text tokens. 
- SimVTP not only benefits from a high video masking ratio (e.g. 90%) due to the temporal redundancy of video, but also needs a high text masking ratio (e.g. 75%), which is much higher than BERT (e.g. 15%), to achieve optimal performance.  
- Equipping SimVTP with video-text contrastive learning (VTC) and video-text matching (VTM), which are two commonly used cross-modal training strategies, could further improve the transferable performance significantly. 
- SimVTP is data-efficient, e.g., pre-training only on 10% data of WebVid-2M, SimVTP achieves surprisingly good results (43.8 R@1) on MSRVTT, which is far above recent state-of-the-art methods pre-trained on both CC3M and WebVid-2M.  

![teaser](imgs/framework.png)


## 🔥 Main Results on Downstream Tasks
### Text-to-video Retrieval on MSR-VTT


|  Method   | Vis Enc.Init | Pre-trained Data |  #pairs |  R@1 | R@5 | R@10 | MdR |
| :------:  | :------:     | :---:            | :---:   | :-----: | :---: | :---: |:---: | 
| HERO  | ImageNet, Kinetics | HowTo100M   | 136M   | 16.8 | 43.4 | 57.7 | - | 
| AVLnet  | ImageNet, Kinetics | HowTo100M   | 136M   | 27.1 | 55.6 | 66.6 | 4 | 
| Frozen  | ImageNet           | WebVid2M+CC3M   | 5.5M   | 31.0 | 59.5 | 70.5 | 3 | 
| OATrans  | ImageNet           | WebVid2M+CC3M   | 5.5M   | 35.8 | 63.4 | 76.5 | 3 | 
| RegionLearner  | ImageNet           | WebVid2M+CC3M   | 5.5M   | 36.3 | 63.9 | 72.5 | 3 | 
| LocVTP  | ImageNet           | WebVid2M+CC3M   | 5.5M   | 36.5 | 64.3 | 76.8 | 3 | 
| BFormer  | ImageNet           | WebVid2M+CC3M   | 5.5M   | 37.6 | 64.8 | 75.1 | 3 | 
| **SimVTP(ours)**  | Kinetics           | WebVid2M   | 2.5M   | **53.6** | **82.8** | **90.8** | **1** | 



## 🔨 Dependencies and Installation


- Python >= 3.6 
- PyTorch >= 1.6.0
- NVIDIA GPU + CUDA

### ⛺ Installation
1. Clone repo
    ```bash
    git clone git@github.com:mayuelala/SimVTP.git
    cd SimVTP
    ```
2. Install dependent packages
    ```bash
    pip install -r requirements.txt
    ```

## 🔅 Data Preparation
Please refer to [`DATA.md`](DATA.md)  for pre-training and downstream evaluation datasets.

## 🌿 Pre-training
We pretrain our SimVTP on video dataset WebVid-2M with 64 V100 GPU (8 nodes x 8 GPUs). The implementation of our SimVTP supports multi-node distributed training. We provide the scripts in the [scripts folder](scripts). 

```bash
bash scripts/pretrain_webvid.sh
```
you could run the scripts respectively. `--master_addr` is set as the ip of the node 0 and `--node_rank` is set from 0 to 7.


## 🍄 Fine-tuning on MSRVTT
We finetune our SimVTP on MSRVTT with 8 V100. We provide the scripts in the [scripts folder](scripts). 
```bash
bash scripts/finetune_msrvtt.sh
``` 
You could also add the `--only_test` to evaluate our finetuned model.

## 🐧  Model Weight
We provide the pretrained weights and finetuned weight on msrvtt in google driver.

|  Method   | Backbone | Epoch |  Pre-train |   Fine-tune | R@1 |
| :------:  | :------: | :---: | :---: | :-----: | :---: | 
| SimVTP    | ViT-B    | 200   | [script](scripts/pretrain_webvid.sh)/[log](https://drive.google.com/drive/folders/1ln0ISwm6y12bUKxFH6AJnWDdE9Llj74Y?usp=share_link)/[checkpoint](https://drive.google.com/drive/folders/1ln0ISwm6y12bUKxFH6AJnWDdE9Llj74Y?usp=share_link) | [script](scripts/finetune_msrvtt.sh)/[log](https://drive.google.com/drive/folders/1ln0ISwm6y12bUKxFH6AJnWDdE9Llj74Y?usp=share_link)/[checkpoint](https://drive.google.com/drive/folders/1ln0ISwm6y12bUKxFH6AJnWDdE9Llj74Y?usp=share_link) | 53.6 |

 ## 👀 Visualization
We provide the script for visualization in [vis.sh](scripts/vis.sh). Though not the exact same as original texts, the reconstructed texts are plausible and in harmony with the video content. Sometimes, they are even more accurate than original texts, like the `white cat` and `little boy` in the second and third columns

![teaser](imgs/vis.png)
## 🔒 License
The majority of this project is released under the CC-BY-NC 4.0 license as found in the LICENSE file.

## 👏 Acknowledgement
This project is built upon [MAE-pytorch](https://github.com/pengzhiliang/MAE-pytorch) and [VideoMAE](https://github.com/MCG-NJU/VideoMAE). Thanks to the contributors of these great codebases.

## ✏️ Citation
```bash
@article{ma2022simvtp,
  title={SimVTP: Simple Video Text Pre-training with Masked Autoencoders},
  author={Ma, Yue and Yang, Tianyu and Shan, Yin and Li, Xiu},
  journal={arXiv preprint arXiv:2212.03490},
  year={2022}
}
``` 
