import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from pathlib import Path

def getmeanstd(dataroot, img_h, img_w,batch_size):
    transform = transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor()
    ])
    dataset = ImageFolder(dataroot, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialize the variables for calculating mean and std
    mean = torch.zeros(3)
    std = torch.zeros(3)
    num_images = 0

    # Calculate the mean for each channel
    for images, _ in data_loader:
        num_images += images.size(0)
        mean += torch.mean(images, dim=(0, 2, 3))

    mean /= num_images

    # Calculate the standard deviation for each channel
    for images, _ in data_loader:
        std += torch.mean((images - mean.view(1, 3, 1, 1)) ** 2, dim=(0, 2, 3))

    std = torch.sqrt(std / num_images)
    print("Dataset: ", dataroot)
    print("Mean values: ", mean)
    print("Standard deviation values: ", std)
    return mean, std


def check_create_dir(path):
    directory = Path(path)
    if not directory.exists():
        try:
            directory.mkdir(parents=True, exist_ok=False)
            print(f"Directory '{path}' created successfully.")
        except OSError as error:
            print(f"Error creating directory '{path}': {error}")
    else:
        print(f"Directory '{path}' already exists.")