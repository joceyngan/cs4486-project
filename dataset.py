import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image

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