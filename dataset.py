from PIL import Image
import torch
from torch.utils.data import Dataset
import os

class ChestXRayDataset(Dataset):
    def __init__(self, data, labels, image_files, metadata, transform=None):
        self.data = data
        self.labels = labels
        self.image_files = image_files
        self.metadata = metadata
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = next((img for img in self.image_files if os.path.basename(img) == row.name), None)
        image = Image.open(img_path).convert('L')
        label = self.labels[idx]
        metadata = torch.tensor(self.metadata.iloc[idx].values, dtype=torch.float)

        if self.transform:
            image = self.transform(image)

        return image, label, metadata