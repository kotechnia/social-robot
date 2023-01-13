# Social Robot Customer Service
## Directories
```
📂 social-robot
├─ 📂data
│  ├─ 📂annotations
│  └─ 📂videos
├─ 📂pdvc 
├─ 📂transformer
├─ 📂emotion 
├─ 📄requirements.txt
├─ 📄prepare_common_dataset.py
├─ 📉dataset_split_list.csv
└─ 📄README.md
```


## Contents 
1. [Common](#Common)
3. [PDVC](#PDVC)
4. [Transformer](#Transformer)
5. [Emotion](#PDVC)

     
     
## Common
> Subtask models of social robot customer service.
The following `python prepare_common_dataset.py` is executed to use a common dataset.
```
$ python prepare_common_dataset.py \
--video_root data/videos \
--annotation_root data/annotations \
--dataset_split_list ../dataset_split_list.csv 
```
    
     
     
## PDVC
### 1. Feature Extraction
```
pdvc $ python extract_feature.py \
--video_root ../data/videos \
--feature_root ../data/features \
--num_workers 20
```
### 2. Preparation
```
pdvc $ python prepare.py \
--dataset_split_list ../dataset_split_list.csv \
--video_root ../data/videos \
--json_root ../data/annotations \
--feature_root ../data/features \
--base_cfg_path cfgs/social_robot.yml \
--cfg_path cfgs/social_robot.yml \
--pdvc_dataset_path data/social_robot.json
```
### 3. Training
```
pdvc $ python train.py \
--cfg_path cfgs/social_robot.yml \
--gpu_id 0 1
```
### 4. Evaluation
```
pdvc $ python eval.py \
--eval_folder social_robot \
--gpu_id 0 1
```
     
     
## Transformer
### 1. Preparation
```
transformer $ python prepare.py \
--dataset_split_list ./../dataset_split_list.csv \
--annotation_json_root ./../data/annotations/ \
--transformer_dataset transformer_dataset.csv
```
### 2. Tokenization
```
transformer $ python tokenizer.py \
--dataset transformer_dataset.csv \
--tokenizer tokenizer \
--temp_dir data \
--vocap_size 16000 \
--limit_alphabet 8000
```
### 3. Training
```
transformer $ python train.py \
--dataset transformer_dataset.csv \
--model transformer_model.h5 \
--tokenizer tokenizer \
--epoch 60 \
--batch_size 256 \
--max_length 80 \
--buffer_size 200000
```
### 4. Evaluation
```
transformer $ python eval.py \
--dataset transformer_dataset.csv \
--model transformer_model.h5 \
--tokenizer tokenizer \
--results_path results.csv \
--max_length 80 \
--buffer_size 200000
```
    
     
     
## Emotion
### 1. Preparation
```
emotion $ python emotion_prepare.py \
--dataset_split_list ../dataset_split_list.csv \
--video_root ../data/videos \
--annotation_root ../data/annotations \
--image_root ../data/faces \
--dataset_path emotion_dataset.csv \
--extract_image
```

### 2. Training
```
emotion $ python train.py \
--dataset emotion_dataset.csv \
--model models/emotion_model.pt \
--fine_epochs 3 \
--epochs 6 \
--batch_size 64 \
--num_workers 4
```

### 3. Evaluation
```
emotion $ python eval.py \
--dataset emotion_dataset.csv \
--model models/emotion_model.pt \
--results_path results.csv \
--batch_size 64 \
--num_workers 4
```


## Original and Reference 
- [PDVC](https://github.com/ttengwang/PDVC) github
- [Transformer](https://github.com/ukairia777/tensorflow-transformer) github
- [Emotion](https://github.com/HSE-asavchenko/face-emotion-recognition) github

