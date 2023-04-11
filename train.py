import os
import pathlib
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import torchvision
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
from models import CNN, VGG
from utils import getmeanstd, check_create_dir
from datetime import datetime
import random
import wandb

dataroot = './Topic_5_Data/ISIC84by84'  #change to your data root dir
train_data_dir = pathlib.Path(dataroot+'/Train')
test_data_dir = pathlib.Path(dataroot+'/Test')

modelroot = './models/'  # your trained models will be saved here
check_create_dir(modelroot)

datetime_str = datetime.now().strftime("%Y%m%d%H%M%S")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32
img_h = 84
img_w = 84

class ISICDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = []

        for label, class_name in enumerate(sorted(os.listdir(data_dir))):
            if not os.path.isdir(os.path.join(data_dir, class_name)):
                continue
            class_dir = os.path.join(data_dir, class_name)
            filenames = os.listdir(class_dir)
            for filename in filenames:
                self.images.append(os.path.join(class_dir, filename))
                self.labels.append(label)
            self.class_names.append(class_name)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Data augmentation and normalization for training
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.RandomResizedCrop(84, scale=(0.9, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.0209, 0.0166, 0.0164], [0.1211, 0.0976, 0.0974]) # prev calculated mean & std of topic 5 train set
        # transforms.Normalize(getmeanstd(train_data_dir,img_h, img_w,batch_size)[0], getmeanstd(train_data_dir,img_h, img_w,batch_size)[1])
    ]),
    'test': transforms.Compose([
        transforms.Resize(84),
        transforms.ToTensor(),
        transforms.Normalize([0.0201, 0.0162, 0.0163], [0.1177, 0.0961, 0.0972]) # prev calculated mean & std of topic 5 test set
        # transforms.Normalize(getmeanstd(test_data_dir, img_h, img_w,batch_size)[0], getmeanstd(test_data_dir,img_h, img_w,batch_size)[1])
    ]),
}

train_dataset = ISICDataset(train_data_dir, transform=data_transforms['train'])
test_dataset = ISICDataset(test_data_dir, transform=data_transforms['test'])

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
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False, num_workers=4)
val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Generate a random seed
seed = random.randint(0, 2**32 - 1)

# Set the seed for PyTorch, random, and CUDA 
torch.manual_seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# training configs
phase = 'ft' #['tr', 'ft']
# phase = 'tr', models=['CNN', 'VGG']
# phase = 'ft', models=['resnet50', 'resnet152', 'densenet121']
model_name = 'convnext_large' 
num_epochs = 100
optimizer_name = 'AdamW'
learning_rate = 0.0001
weight_decay = 1e-4
class_weighting = False
best_val_acc = 0.0
goal_accu = 0.85
dropout = 0.2

num_classes = len(train_dataset.class_names)
  
#select model
def choose_model(model_name):
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
        }.get(model_name,models.resnet50(pretrained=True))

model = choose_model(model_name)
if phase =='ft':
    seed = 'NIL'
    if model_name in ['resnet50', 'resnet152', 'densenet121', 'efficientnet_b7','inception_v3','convnext_large']:
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_name == 'mobilenet_v2':
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
model = model.to(device)

class_weights = {'AK': 4.1665, 'BCC': 0.95016, 'BKL': 1.2133, 'DF': 22.03147, 'MEL': 0.69253, 'NV': 0.23972, 'SCC': 5.79995, 'VASC': 20.01552}
weights = [class_weights[class_name] for class_name in train_dataset.class_names]
weights = torch.tensor(weights, device=device, dtype=torch.float)

#select loss function
if class_weighting:
    criterion = nn.CrossEntropyLoss(weight=weights)  # use class-weighting
else:
    criterion = nn.CrossEntropyLoss()

#select optimiser
def choose_optimizer(optimizer_name):
    return {
        'Adam': optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay),
        'AdamW': optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay),
        'SGD': optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9),
        'Adagrad': optim.Adagrad(model.parameters(), lr=learning_rate)
    }.get(optimizer_name,optim.Adam(model.parameters(), lr=learning_rate))   
optimizer = choose_optimizer(optimizer_name)

train_name = f'{model_name}-{phase}-'+datetime_str

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="cs4486-hw3-skin-cancer-classification",
    name=train_name,
    # track hyperparameters and run metadata
    config={
    "architecture": model_name,
    "optimizer": optimizer_name,
    "dataset": "ISIC84by84",
    "epochs": num_epochs,
    "seed": seed,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "class_weighting": class_weighting,
    "dropout": dropout,
    }
)

# for visualisation
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

#for early stopping params
patience = 10
min_delta = 0.001
moving_average_alpha = 0.5  # value to control the smoothness of the moving average
best_val_loss = float("inf")
counter = 0
moving_average_loss = None

#training loop
history_folder = "./results/{}/train_history/".format(train_name)
check_create_dir(history_folder)

for epoch in range(num_epochs):
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

    if epoch_acc_val > best_val_acc and epoch != 0 and epoch_acc_val > goal_accu:
        best_val_acc = epoch_acc_val
        torch.save(model.state_dict(), f"{modelroot}{datetime_str}_{model_name}_ep{epoch+1}_acc{epoch_acc_val:.2f}.pth")
    if (epoch+1) % 20 == 0:
        torch.save(model.state_dict(), f"{modelroot}{datetime_str}_{model_name}_ep{epoch+1}_acc{epoch_acc_val:.2f}.pth")

    #early stopping implenmentation
    if moving_average_loss is None:
        moving_average_loss = epoch_loss_val
    else:
        moving_average_loss = moving_average_alpha * moving_average_loss + (1 - moving_average_alpha) * epoch_loss_val
    print(f"Epoch {epoch+1}, Validation Loss: {epoch_loss_val:.4f}, Moving Average Loss: {moving_average_loss:.4f}")

    if moving_average_loss < best_val_loss - min_delta:
        best_val_loss = moving_average_loss
        counter = 0
        # Save the model so far
        torch.save(model.state_dict(), f"{modelroot}{datetime_str}_{model_name}_ep{epoch+1}_acc{epoch_acc_val:.2f}.pth")
    else:
        counter += 1
        print(f"Early stopping counter: {counter}/{patience}")
        if counter >= patience:
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
