import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from pathlib import Path

from models import CNN, VGG
from utils import check_create_dir
from dataset import ISICDataset

def choose_model(model_name, num_classes):
    if model_name == 'CNN':
        model = CNN(num_classes=num_classes)
    else:
        model = {
            'VGG': VGG(num_classes=num_classes),
            'resnet50': models.resnet50(pretrained=False),
            'resnet152': models.resnet152(pretrained=False),
            'densenet121': models.densenet121(pretrained=False),
            'mobilenet_v2': models.mobilenet_v2(pretrained=False),
            'efficientnet_b7': models.efficientnet_b7(pretrained=False),
            'inception_v3': models.inception_v3(pretrained=False),
            'swin_v2_b': models.swin_v2_b(pretrained=False),
            'vit_l_32': models.vit_l_32(pretrained=False),
            'convnext_large':models.convnext_large(pretrained=False)
        }.get(model_name, print(""))

    # Update the output layers for other models
    if model_name in ['resnet50', 'resnet152', 'densenet121', 'efficientnet_b7','inception_v3']:
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_name == 'mobilenet_v2':
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    elif model_name == 'convnext_large':
        num_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_features, num_classes)
    elif model_name in ['vit_l_32', 'swin_v2_b']:
        num_features = model.head.in_features
        model.head = nn.Linear(num_features, num_classes)

    return model


def predict(model, test_data):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in test_data:
            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels

def plot_confusion_matrix(y_true, y_pred, class_names, train_name):
    cm_folder = './cm/'
    check_create_dir(cm_folder)
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

    # Normalize the confusion matrix to get the accuracy per class
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    df_cm_normalized = pd.DataFrame(cm_normalized, index=class_names, columns=class_names)

    # Custom cell label format to display count and accuracy
    cell_labels = np.array([["{}\n{:.1%}".format(count, acc) for count, acc in zip(row_counts, row_accs)]
                            for row_counts, row_accs in zip(cm, cm_normalized)])

    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=cell_labels, cmap="RdPu", fmt='', cbar=False)
    plt.title(filename)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(cm_folder+"{}_cm.png".format(train_name))
    # plt.show()

if __name__ == "__main__":
    dataroot = './Topic_5_Data/ISIC84by84'  #change to your data root dir
    test_data_dir = Path(dataroot+'/Test')
    data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(84),
        transforms.ToTensor(),
        transforms.Normalize([0.0201, 0.0162, 0.0163], [0.1177, 0.0961, 0.0972]) # prev calculated mean & std of topic 5 test set
        # transforms.Normalize(getmeanstd(test_data_dir, img_h, img_w,batch_size)[0], getmeanstd(test_data_dir,img_h, img_w,batch_size)[1])
        ]),
    }
    test_dataset = ISICDataset(test_data_dir, transform=data_transforms['test'])
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./models/20230411233204_convnext_large_ep18_acc0.82.pth"
    filename = model_path.split("/")[-1]  # Get the filename without the directory path
    train_name = filename.split("_")[0]
    model_name = "convnext_large"
    
    test_data = test_loader
    class_names = test_dataset.class_names

    # Replace with your own model and test data
    model = choose_model(model_name, len(test_dataset.class_names))
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    predictions, true_labels = predict(model, test_data)
    plot_confusion_matrix(true_labels, predictions, class_names, filename)
