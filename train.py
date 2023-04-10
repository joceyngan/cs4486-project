import os
import pathlib
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import datasets, transforms
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

dataroot = './data/ISIC84by84' #change to your data root dir
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
        transforms.RandomRotation(20),
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

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
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
num_epochs = 50
model_name = 'VGG'
optimizer_name = 'AdamW'
learning_rate = 0.001
weight_decay = 1e-4
best_val_acc = 0.0
goal_accuracy = 0.90
dropout = 0.2


num_classes = len(train_dataset.class_names)
  
#select model
def choose_model(model_name):
    return {
        'CNN': CNN(num_classes=num_classes, dropout=dropout).to(device),
        'VGG': VGG(num_classes=num_classes, dropout=dropout).to(device)
    }.get(model_name,9)
model = choose_model(model_name)

#select loss function
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

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="cs4486-hw3-skin-cancer-classification",
    name=datetime_str,
    # track hyperparameters and run metadata
    config={
    "architecture": model_name,
    "optimizer": optimizer_name,
    "dataset": "ISIC84by84",
    "epochs": num_epochs,
    "seed": seed,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "dropout": dropout,
    }
)

# for visualisation
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

#training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc.item())
    # log metrics to wandb
    wandb.log({"train_accuracy": epoch_acc, "train_loss": epoch_loss})
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}")

    model.eval()
    running_loss_val = 0.0
    running_corrects_val = 0
    for inputs_val, labels_val in test_loader:
        inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
        with torch.no_grad():
            outputs_val = model(inputs_val)
            _, preds_val = torch.max(outputs_val, 1)
            loss_val = criterion(outputs_val, labels_val)

        running_loss_val += loss_val.item() * inputs_val.size(0)
        running_corrects_val += torch.sum(preds_val == labels_val.data)

    epoch_loss_val = running_loss_val / len(test_dataset)
    epoch_acc_val = running_corrects_val.double() / len(test_dataset)
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

    if epoch_acc_val > best_val_acc and epoch != 0 and epoch_acc_val > 0.3:
        best_val_acc = epoch_acc_val
        torch.save(model.state_dict(), f"{modelroot}{datetime_str}_{model_name}_ep{epoch+1}_acc{epoch_acc_val:.2f}.pth")
    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), f"{modelroot}{datetime_str}_{model_name}_ep{epoch+1}_acc{epoch_acc_val:.2f}.pth")


# Plot train and validation losses
history_folder = "./results/{}/train_history/".format(datetime_str)
check_create_dir(history_folder)
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("")
plt.legend()
plt.savefig(history_folder+"{}_training_history.png".format(datetime_str))
# plt.show()

test_data = torchvision.datasets.ImageFolder(
    test_data_dir,
    transform=transforms.Compose([transforms.Resize((img_h, img_w)),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.0201, 0.0162, 0.0163], [0.1177, 0.0961, 0.0972])
                                #   transforms.Normalize(getmeanstd(test_data_dir,img_h, img_w,batch_size)[0], getmeanstd(test_data_dir,img_h, img_w,batch_size)[1])
                                ]))
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=4)

class_names = test_data.classes

# Visualize some test images and their predicted labels
# num_images_to_show = 10
# output_folder = "./results/{}/output_images/".format(datetime_str)
# check_create_dir(output_folder)
# class_names = test_data.classes

# for i, (inputs, targets) in enumerate(test_loader):
#     if i >= num_images_to_show:
#         break
#     inputs = inputs.to(device)
#     targets = targets.to(device)
#     outputs = model(inputs)
#     _, predicted_labels = torch.max(outputs, 1)
#     images = inputs.cpu().numpy().transpose((0, 2, 3, 1))
#     true_labels = targets.cpu().numpy()

#     for j in range(inputs.shape[0]):
#         plt.figure()
#         plt.imshow(images[j])
#         plt.title(f"True Label: {class_names[true_labels[j]]}, Predicted Label: {class_names[predicted_labels[j]]}")
#         plt.savefig(f"{output_folder}/image_{i * inputs.shape[0] + j}.jpg")

