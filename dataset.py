from PIL import Image
import os
from torch.utils.data import Dataset


class ChestXRayDataset(Dataset):
    def __init__(self, dataframe, labels, image_files, transform=None):
        self.dataframe = dataframe
        self.labels = labels
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_index = row['Image Index']

        try:
            image_path = next(path for path in self.image_files if os.path.basename(path) == image_index)
        except StopIteration:
            raise FileNotFoundError(f"Image file not found for index: {image_index}")

        image = Image.open(image_path).convert('L')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label