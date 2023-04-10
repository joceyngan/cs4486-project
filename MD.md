For PyTorch version
1. install requirements.txt "For PyTorch version" part for most updated codes
2. will need an wandb account to be able to visualise progress in real time, if dont have wandb acccount can comment out all wandb code in train.py
3. training configuration is inside the train.py, will modify to be able to use command line to change config if have time later
4. change the dataroot in train.py according to your dataset path
5. for convenience, normalisation using pre-calculated mean and std to save time, if dataset has any change, call the getmeanstd method in utils.py to calculate the updated mean and std for data normalisation. Code is comment out right now in train.py. Can switch easily if needed.

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

run
 ```CUDA_LAUNCH_BLOCKING=1 python train.py ```