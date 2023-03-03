



## 🍀 DATA  Preparation

### Pre-training
1. Download [WebVid-2M](https://github.com/m-bain/webvid)
2. Gererate a json file from WebVid-2M annotation. We provide the pretraining annotation [here](https://drive.google.com/drive/folders/1ln0ISwm6y12bUKxFH6AJnWDdE9Llj74Y?usp=share_link)
```bash
    {
        YOUR VIDEO PATH: CAPTION
        YOUR VIDEO PATH: CAPTION
        YOUR VIDEO PATH: CAPTION
        ....
        YOUR VIDEO PATH: CAPTION
        YOUR VIDEO PATH: CAPTION
    }
```

### Fine-tuning
1. Download MSRVTT
```bash
mkdir data
cd data
wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip  
unzip MSRVTT.zip 
```
1. Split the train/test set following the [Frozen in time](https://github.com/m-bain/frozen-in-time). Gererate the train/test json file from MSRVTT. We provide our splited annotation [here](https://drive.google.com/drive/folders/1ln0ISwm6y12bUKxFH6AJnWDdE9Llj74Y?usp=share_link)

```bash
    {
        YOUR VIDEO PATH: CAPTION
        YOUR VIDEO PATH: CAPTION
        YOUR VIDEO PATH: CAPTION
        ....
        YOUR VIDEO PATH: CAPTION
        YOUR VIDEO PATH: CAPTION
    }
```