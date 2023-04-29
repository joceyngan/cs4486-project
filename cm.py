import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from pathlib import Path
from tqdm import tqdm
from models import CNN, VGG
from utils import getmeanstd, check_create_dir
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
    elif model_name in ['swin_v2_b']:
        num_features = model.head.in_features
        model.head = nn.Linear(num_features, num_classes)
    elif model_name in ['vit_l_32']:
        num_features = model.heads.head.in_features
        model.heads.head = nn.Linear(num_features, num_classes)

    return model


def predict(model, test_data):
    model.eval()
    all_preds = []
    all_topk_preds = {k: [] for k in range(1, 6)} 
    all_labels = []

    with torch.no_grad():
        for data, labels in tqdm(test_data):
            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)
            _, preds = torch.max(outputs, 1)

            for k in range(1, 6):
                _, topk_preds = torch.topk(outputs, k, 1)
                for i in range(len(topk_preds)):
                    all_topk_preds[k].append(topk_preds[i].cpu().numpy())

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_topk_preds, all_labels 

def top_k_accuracy(y_true, y_topk_preds, k=1):
    count = 0
    for i in range(len(y_true)):
        if y_true[i] in y_topk_preds[i][:k]:
            count += 1
    return count / len(y_true)


def plot_confusion_matrix(y_true, y_pred, class_names, filename, top1_acc, top2_acc, top3_acc):
    cm_folder = './results/cm/'
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
    plt.title(filename, y=1.08)
    plt.text(0.5, 1.02, "Top-1 Acc: {:.2%} | Top-2 Acc: {:.2%} | Top-3 Acc: {:.2%}".format(top1_acc, top2_acc, top3_acc),
             horizontalalignment='center',
             fontsize=12,
             transform=plt.gca().transAxes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(cm_folder+"{}_cm.png".format(filename))
    # plt.show()
    
def plot_f1(class_names, f1_scores_per_class, avg_f1_score):
    f1_folder = './results/f1/'
    check_create_dir(f1_folder)
    
    data = {"Class": class_names, "F1 Score": f1_scores_per_class}
    df = pd.DataFrame(data)
    df.loc[len(df)] = ["Average", avg_f1_score]

    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    ax = sns.barplot(x="Class", y="F1 Score", data=df, palette="coolwarm")
    ax.set(ylim=(0, 1))  # Set y-axis limit to range from 0 to 1
    plt.title("{}\nF1 Scores per Class and Average F1 Score".format(filename))

    # Annotate the bars with their respective values
    for p in ax.patches:
        ax.annotate(
            format(p.get_height(), ".2f"),
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="baseline",
            fontsize=12,
            color="black",
            xytext=(0, 5),
            textcoords="offset points",
    )
    plt.savefig(f1_folder+"{}_f1.png".format(filename))
    # plt.show()

if __name__ == "__main__":
    model_path = "./models/20230429032042_CNN_ep410_acc0.81.pth" # change to your model path
    model_name = "CNN"                                           # change this too
    filename = model_path.split("/")[-1]
    train_name = filename.split("_")[0]
    dataroot = './Topic_5_Data/ISIC84by84'  #change to your data root dir
    test_data_dir = Path(dataroot+'/Test')
    img_h = img_w = 84
    batch_size = 64
    data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(img_h),
        transforms.ToTensor(),
        transforms.Normalize([0.0104, 0.0084, 0.0085], [0.0858, 0.0702, 0.0710]) # prev calculated mean & std of topic 5 test set
        # transforms.Normalize(getmeanstd(test_data_dir, img_h, img_w,batch_size)[0], getmeanstd(test_data_dir,img_h, img_w,batch_size)[1])
        ]),
    }
    test_dataset = ISICDataset(test_data_dir, transform=data_transforms['test'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_data = test_loader
    print('model_path: ', model_path)
    print("Length of test_data:", len(test_data.dataset))
    class_names = test_dataset.class_names

    # Replace with your own model and test data
    model = choose_model(model_name, len(test_dataset.class_names))
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    predictions, all_topk_predictions, true_labels = predict(model, test_data)
    
    # Calculate the F1 score for each class
    f1_scores_per_class = f1_score(true_labels, predictions, average=None)
    # Calculate the average F1 score (macro-average)
    avg_f1_score = f1_score(true_labels, predictions, average='macro')

    print("F1 scores per class:", f1_scores_per_class)
    print("Average F1 score:", avg_f1_score)

    accuracy_scores = {}
    for k in range(1, 6):
        accuracy_scores[f"top{k}_accuracy"] = top_k_accuracy(true_labels, all_topk_predictions[k], k=k)

    print(accuracy_scores)

    top1_accuracy = accuracy_scores['top1_accuracy']
    top2_accuracy = accuracy_scores['top2_accuracy']
    top3_accuracy = accuracy_scores['top3_accuracy']
    plot_confusion_matrix(true_labels, predictions, class_names, filename, top1_accuracy, top2_accuracy, top3_accuracy)
    plot_f1(class_names, f1_scores_per_class, avg_f1_score)