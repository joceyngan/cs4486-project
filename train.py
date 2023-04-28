import pathlib
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler, WeightedRandomSampler
from torchvision import transforms, models
import torchvision.transforms as transforms
from tqdm import tqdm
from models import CNN, VGG
from utils import getmeanstd, check_create_dir
from datetime import datetime
import random
import wandb
from dataset import ISICDataset
import time
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np

dataroot = './Topic_5_Data/ISIC84by84_new'  #change to your data root dir
train_data_dir = pathlib.Path(dataroot+'/Train')
test_data_dir = pathlib.Path(dataroot+'/Test')

modelroot = './models/'  # your trained models will be saved here
check_create_dir(modelroot)

datetime_str = datetime.now().strftime("%Y%m%d%H%M%S")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
img_h = 84
img_w = 84

# ./Topic_5_Data/ISIC84by84/Train mean and std:
# ([0.0209, 0.0166, 0.0164], [0.1211, 0.0976, 0.0974])

# ./Topic_5_Data/ISIC84by84_new/Train mean and std:
# ([0.0105, 0.0083, 0.0082],[0.0870, 0.0700, 0.0699])

# Data augmentation and normalization for training
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.RandomResizedCrop(img_h, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.0105, 0.0083, 0.0082], [0.0870, 0.0700, 0.0699]) # prev calculated mean & std of topic 5 train set
        # transforms.Normalize(getmeanstd(train_data_dir,img_h, img_w,batch_size)[0], getmeanstd(train_data_dir,img_h, img_w,batch_size)[1])
    ]),
    'test': transforms.Compose([
        transforms.Resize(img_h),
        transforms.ToTensor(),
        transforms.Normalize([0.0104, 0.0084, 0.0085], [0.0858, 0.0702, 0.0710]) # prev calculated mean & std of topic 5 test set
        # transforms.Normalize(getmeanstd(test_data_dir, img_h, img_w,batch_size)[0], getmeanstd(test_data_dir,img_h, img_w,batch_size)[1])
    ]),
}

train_dataset = ISICDataset(train_data_dir, transform=data_transforms['train'])
test_dataset = ISICDataset(test_data_dir, transform=data_transforms['test'])

num_classes = len(train_dataset.class_names)

ttl = len(list(train_data_dir.glob('**/*.jpg')))
counts = {}
for i in range(num_classes):
    image_count = len(list(train_data_dir.glob(train_dataset.class_names[i] + '/*.jpg')))
    counts[train_dataset.class_names[i]] = image_count
    print(train_dataset.class_names[i], ':', image_count)

# Calculate inverted class weights
total_instances = sum(counts.values())
class_weights = {label: total_instances / count for label, count in counts.items()}
print('total_instances: ', total_instances)
print('class_weights: ', class_weights)

# Train & val set split
def stratified_split(labels, train_ratio, seed=None):
    if seed is not None:
        random.seed(seed)
    
    class_indices = {}
    for idx, label in enumerate(labels):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    train_indices = []
    val_indices = []

    for label, indices in class_indices.items():
        split_point = int(train_ratio * len(indices))
        random.shuffle(indices)
        train_indices.extend(indices[:split_point])
        val_indices.extend(indices[split_point:])

    random.shuffle(train_indices)
    random.shuffle(val_indices)
    return train_indices, val_indices

train_indices, val_indices = stratified_split(train_dataset.labels, train_ratio=0.8, seed=42)
# train_sampler = SubsetRandomSampler(train_indices)
# val_sampler = SubsetRandomSampler(val_indices)

# train_weights = compute_sample_weight('balanced', train_dataset.labels, indices=train_indices)
# print('train_weights:', dict(zip(np.unique(train_dataset.labels), train_weights)))
# weighted_sampler = WeightedRandomSampler(weights=class_weights, num_samples=len(instance_weights), replacement=True)

class_weights = {i: total_instances / counts[train_dataset.class_names[i]] for i in range(num_classes)}
print('class_weights: ', class_weights)

# Compute the sample weights for each sample in the train dataset
sample_weights_train = [class_weights[train_dataset.labels[i]] for i in train_indices]
train_sampler = WeightedRandomSampler(sample_weights_train, num_samples=len(train_indices), replacement=True)

# Compute the sample weights for each sample in the val dataset
sample_weights_val = [class_weights[train_dataset.labels[i]] for i in val_indices]
val_sampler = WeightedRandomSampler(sample_weights_val, num_samples=len(val_indices), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False, num_workers=4)
val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Generate a random seed
seed = int(time.time()) % (2**32 - 1)

# Set the seed for PyTorch, random, and CUDA 
torch.manual_seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# training configs
phase = 'tr' #['tr', 'ft']
# phase = 'tr', models=['CNN', 'VGG']
# phase = 'ft', models=['resnet50', 'resnet152', 'densenet121','mobilenet_v2',
#                       'efficientnet_b7','inception_v3','convnext_large',
#                       'vit_l_32', 'swin_v2_b']
model_name = 'CNN'
num_epochs = 500
optimizer_name = 'AdamW' # ['Adam', 'AdamW', 'SGD', 'Adagrad']
learning_rate = 1e-5
weight_decay = 1e-5
class_weighting = True
best_val_acc = 0.0
goal_accu = 0.95
dropout = 0.5

#finetuning configs
resume_training = False
pretrained_model_path = Path('')
start_epoch = 122

# early stopping configs
patience = 20
min_delta = 0.001
moving_average_alpha = 0.75  # close to 1: more forgiving to fluctuations, less sensitive to short-term changes
best_val_loss = float("inf")
counter = 0
moving_average_loss = None
  
#select model
def choose_model(model_name, phase):
    if phase == 'tr':
        return {
            'CNN': CNN(num_classes=num_classes, dropout=dropout),
            'VGG': VGG(num_classes=num_classes, dropout=dropout)
        }.get(model_name,CNN(num_classes=num_classes, dropout=dropout))
    elif phase =='ft':
        return {
            'resnet50': models.resnet50(pretrained=True),
            'resnet152': models.resnet152(pretrained=True),
            'densenet121': models.densenet121(pretrained=True),
            'mobilenet_v2': models.mobilenet_v2(pretrained=True),
            'efficientnet_b7': models.efficientnet_b7(pretrained=True),
            'inception_v3': models.inception_v3(pretrained=True),
            'convnext_large': models.convnext_large(pretrained=True),
            'swin_v2_b': models.swin_v2_b(pretrained=True),
            'vit_l_32': models.vit_l_32(pretrained=True),  #change image data size to 224
            'CNN': CNN(num_classes=num_classes, dropout=dropout),
            'VGG': VGG(num_classes=num_classes, dropout=dropout)
        }.get(model_name,models.resnet50(pretrained=True))
        

model = choose_model(model_name, phase)
if phase =='ft':
    seed = 'NIL'
    if model_name in ['resnet50', 'resnet152', 'densenet121', 'efficientnet_b7','inception_v3']:
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_name == 'mobilenet_v2':
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    elif model_name == 'convnext_large':
        num_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_features, num_classes)
    elif model_name in ['swin_v2_b']:
        num_features = model.head.in_features
        model.head = nn.Linear(num_features, num_classes)
    elif model_name in ['vit_l_32']:
        num_features = model.heads.head.in_features
        model.heads.head = nn.Linear(num_features, num_classes)
    elif model_name in ['CNN', 'VGG']:
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze the last layer
        for param in model.fc2.parameters():
            param.requires_grad = True


if resume_training:
    model.load_state_dict(torch.load(pretrained_model_path))
    pretrained_model_name = pretrained_model_path.stem
    train_name = f'{model_name}-{phase}-{datetime_str}-fm-{pretrained_model_path.stem[:14]}'
    print(f"Resuming training from epoch {start_epoch}")
    print('model: ', pretrained_model_name)
else:
    train_name = f'{model_name}-{phase}-'+datetime_str
    start_epoch = 0

model = model.to(device)

#select loss function
# criterion = nn.CrossEntropyLoss(weight=weights)  # use class-weighting
criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()

#select optimiser
def choose_optimizer(optimizer_name):
    return {
        'Adam': optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay),
        'AdamW': optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay),
        'SGD': optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9),
        'Adagrad': optim.Adagrad(model.parameters(), lr=learning_rate)
    }.get(optimizer_name,optim.Adam(model.parameters(), lr=learning_rate))   
optimizer = choose_optimizer(optimizer_name)


# start a new wandb run to track this script
wandb.init(
    project="cs4486-hw3-skin-cancer-classification",
    name=train_name,
    # track hyperparameters and run metadata
    config={
    "architecture": model_name,
    "optimizer": optimizer_name,
    "dataset": dataroot.split('/')[-1],
    "ft_from": ('{}'.format(pretrained_model_name) if resume_training else "NIL"),
    "batch_size": batch_size,
    "epochs": num_epochs,
    "seed": seed,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "class_weighting": str(class_weighting),
    "dropout": dropout,
    }
)

# for visualisation
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# logging in local
history_folder = "./results/{}/train_history/".format(train_name)
check_create_dir(history_folder)
with open(history_folder+"{}_log.txt".format(train_name), "a") as f:
    f.write('train_name: {}, \n'.format(train_name))
    f.write('train_weights: {}'.format(class_weights))
    f.write('architecture: {}'.format(model_name))
    f.write('optimizer: {}'.format(optimizer_name))
    f.write('dataset: {}'.format(dataroot.split('/')[-1]))
    f.write('ft_from: {}'.format(pretrained_model_name) if resume_training else "NIL")
    f.write('batch_size: {}'.format(batch_size))
    f.write('epochs: {}'.format(num_epochs))
    f.write('seed: {}'.format(seed))
    f.write('learning_rate: {}'.format(learning_rate))
    f.write('weight_decay: {}'.format(weight_decay))
    f.write('class_weighting: {}'.format(str(class_weighting)))
    f.write('dropout: {}'.format(dropout))

#training loop
for epoch in range(start_epoch, num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(train_indices)
    epoch_acc = running_corrects.double() / len(train_indices)
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc.item())
    # log metrics to wandb
    wandb.log({"train_accuracy": epoch_acc, "train_loss": epoch_loss})
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}")

    model.eval()
    running_loss_val = 0.0
    running_corrects_val = 0
    for inputs_val, labels_val in tqdm(val_loader, desc=f" Validation {epoch+1}/{num_epochs}"):
        inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
        with torch.no_grad():
            outputs_val = model(inputs_val)
            _, preds_val = torch.max(outputs_val, 1)
            loss_val = criterion(outputs_val, labels_val)

        running_loss_val += loss_val.item() * inputs_val.size(0)
        running_corrects_val += torch.sum(preds_val == labels_val.data)

    epoch_loss_val = running_loss_val / len(val_indices)
    epoch_acc_val = running_corrects_val.double() / len(val_indices)
    val_losses.append(epoch_loss_val)
    val_accuracies.append(epoch_acc_val.item())
    wandb.log({"val_accuracy": epoch_acc_val, "val_loss": epoch_loss_val})
    print(f"\tValidation Loss: {epoch_loss_val:.4f}, Validation Accuracy: {epoch_acc_val:.4f}")
    
    wandb.log({datetime_str : wandb.plot.line_series(
                       xs=[i for i in range(num_epochs)],
                       ys=[train_accuracies, val_accuracies, train_losses, val_losses],
                       keys=["train_accuracy", "val_accuracy","train_loss", "val_loss"],
                       title="Accuracy & Loss",
                       xname="epoch")})
    
    save_model_name = f"{modelroot}{datetime_str}_{model_name}_ep{epoch+1}_acc{epoch_acc_val:.2f}.pth"
    if epoch_acc_val > best_val_acc and epoch_acc_val >= 0.75:
        best_val_acc = epoch_acc_val
        torch.save(model.state_dict(), save_model_name)
    if epoch_acc_val > goal_accu:
        torch.save(model.state_dict(), save_model_name)
    if (epoch+1) % 20 == 0:
        torch.save(model.state_dict(), save_model_name)


    #early stopping implenmentation
    if sum(val_accuracies[-10:])/10 - sum(val_accuracies[-50:])/50 > 0.01:
        accu_increasing = True
    else:
        accu_increasing = False
    
    if moving_average_loss is None:
        moving_average_loss = epoch_loss_val
    else:
        moving_average_loss = moving_average_alpha * moving_average_loss + (1 - moving_average_alpha) * epoch_loss_val
    print(f"Epoch {epoch+1}, Validation Loss: {epoch_loss_val:.4f}, Moving Average Loss: {moving_average_loss:.4f}")

    if moving_average_loss < best_val_loss - min_delta or accu_increasing:
        best_val_loss = moving_average_loss
        counter = 0
    else:
        counter += 1
        print(f"Early stopping counter: {counter}/{patience}")
        if counter >= patience:
            torch.save(model.state_dict(), save_model_name)
            print("Early stopping triggered.")
            break

    # Plot train and validation losses
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("")
    plt.title(train_name)
    plt.legend()
    plt.savefig(history_folder+"{}_training_history.png".format(train_name))
    # plt.show()
