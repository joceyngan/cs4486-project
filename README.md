<h2>1. Folder Structure:</h2> 

```
root
└── dataroot
    └── ISIC84by84
        ├── Test
        │   ├── class_1
        │   └── class_2
        └── Train
            ├── class_1
            └── class_2
├── models
    └── YYYYmmddHHMMSS_model_ep_v_acc.pth (model name format)
├── results (for history and test output)
    └── YYYYmmddHHMMSS (training name format)
├── cm.py
├── dataset.py 
├── wandb (if applicable)
├── models.py
├── train.py
└── utils.py
```

<h2>2. Datasets</h2>  



Original ISIC84by84 * : [link](https://portland-my.sharepoint.com/:u:/g/personal/szfung9-c_my_cityu_edu_hk/ERZovvPIWtJHtDzDMr6qjcMB-e-1LD91YhZ2jZXQ9DjjlQ?e=AjCdJ6)

Cleansed ISIC84by84 * : [link](https://drive.google.com/file/d/1WYQ3FPrdfN4c6FLp9_f_wQNZe4D2wHMo/view?usp=sharing)

*Please note the above datasets are provided by CityU for the CS4486 course project, does not guaranteed to be the same as official ISIC dataset.  
 For the official ISIC data please refer to [ISIC Archive](https://www.isic-archive.com/#!/topWithHeader/wideContentTop/main).

<h2>3. Best Models:</h2> 

Best model of each networks are trained/finetuned on original or cleansed dataset

|Network| CNN | VGG19 | Resnet50 | ConvNeXT | Swin Transformer |  
|-------|-----|-------|----------|----------|------------------|  
|Validation accuracy | 74% | 75% | 78% | 85% | 84% |  
|Top 1 accuracy | 56% | 56% | 65% | <mark>73%</mark> | 67% |  

CNN: [link](https://portland-my.sharepoint.com/:u:/g/personal/szfung9-c_my_cityu_edu_hk/ESHombM__v1LiuUuGdtUpLcBB_Xi-ojziah5jdyTchR5Bg?e=2cvISS)

VGG19: [link](https://portland-my.sharepoint.com/:u:/g/personal/szfung9-c_my_cityu_edu_hk/EdxUBmfARNZIpP_CsexoH9AB1dBcGb-KJ8FwuzH0qlgZyQ?e=xsEQ18)

Resnet50  (Finetuned from Imagenet pretrained weight): [link](https://portland-my.sharepoint.com/:u:/g/personal/szfung9-c_my_cityu_edu_hk/ERXEZIn1NehFjOzuGbr7P8ABnt35KIR23vtRiMa-lhXx2A?e=K6A4Bv)

ConvNeXT Large (Finetuned from Imagenet pretrained weight): [link](https://portland-my.sharepoint.com/:u:/g/personal/szfung9-c_my_cityu_edu_hk/EbrnGWzNGdtKtuk_zuGqLv8BGzBG_yDsjH6pBsLGCrPkMA?e=j24kHt)

Swin Transformer v2 (Finetuned from Imagenet pretrained weight): [link](https://portland-my.sharepoint.com/:u:/g/personal/szfung9-c_my_cityu_edu_hk/EW2aLbpq_3pIoZD073MIsyQBZkL0bcxSfIAuMFV-MdyzCQ?e=h4zQ9b)


<h2>5. Training & Inference</h2>  

For model training and finetuning:
train.py
1. change dataroot path 
2. for training modify # training configs 
3. for finetuning also modify #finetuning configs 
4. run ```CUDA_LAUNCH_BLOCKING=1 python train.py ``` 

For ploting confusion matrix and F1 scores:
cm.py
1. change dataroot path
2. change model_path and model_name, model_name same as listed in train.py
3. run ```python cm.py ```
4. if need pure inference can modify from cm.py, due to time limit of project it is not provided.
