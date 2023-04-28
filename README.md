Folder structure:
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
├── wandb (if applicable)
├── models.py
├── train.py
└── utils.py
```

Best models trained/finetuned on "ISIC84by84" dataset:

CNN:
https://portland-my.sharepoint.com/:u:/g/personal/szfung9-c_my_cityu_edu_hk/ESHombM__v1LiuUuGdtUpLcBB_Xi-ojziah5jdyTchR5Bg?e=2cvISS

VGG:
https://portland-my.sharepoint.com/:u:/g/personal/szfung9-c_my_cityu_edu_hk/EdxUBmfARNZIpP_CsexoH9AB1dBcGb-KJ8FwuzH0qlgZyQ?e=xsEQ18

Resnet50  (Finetuned from Imagenet pretrained weight):
https://portland-my.sharepoint.com/:u:/g/personal/szfung9-c_my_cityu_edu_hk/ERXEZIn1NehFjOzuGbr7P8ABnt35KIR23vtRiMa-lhXx2A?e=K6A4Bv

ConvNeXT (Finetuned from Imagenet pretrained weight):
https://portland-my.sharepoint.com/:u:/g/personal/szfung9-c_my_cityu_edu_hk/EbrnGWzNGdtKtuk_zuGqLv8BGzBG_yDsjH6pBsLGCrPkMA?e=j24kHt

Swin Transformer (Finetuned from Imagenet pretrained weight):
https://portland-my.sharepoint.com/:u:/g/personal/szfung9-c_my_cityu_edu_hk/EW2aLbpq_3pIoZD073MIsyQBZkL0bcxSfIAuMFV-MdyzCQ?e=h4zQ9b



For model training and finetuning:
train.py
1. change dataroot path
2. for training modify # training configs
3. for finetuning also modify #finetuning configs
4. run ```CUDA_LAUNCH_BLOCKING=1 python train.py ```

For ploting confusion matrix and F1 scores:
cm.py
1. change dataroot path
2. change model_path and model_name, model_name same as list in train.py
3. run ```python cm.py ```
