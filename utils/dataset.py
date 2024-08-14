import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class CelebaDataset(Dataset):
    def __init__(self, path, size=128, limit=10000):
        self.size = (size, size)
        self.items = []
        self.labels = []

        for data in os.listdir(path)[:limit]:
            item = os.path.join(path, data)
            self.items.append(item)
            self.labels.append(data)

        self.transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        data = Image.open(self.items[idx]).convert('RGB')
        data = self.transform(data)
        return data, self.labels[idx]
