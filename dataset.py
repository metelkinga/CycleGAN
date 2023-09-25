from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np


class nontumorousDataset(Dataset):
    def __init__(self, root_tumorous, root_nontumorous, transform=None):
        self.root_tumorous = root_tumorous
        self.root_nontumorous = root_nontumorous
        self.transform = transform

        self.tumorous_images = os.listdir(root_tumorous)
        self.nontumorous_images = os.listdir(root_nontumorous)
        self.length_dataset = max(len(self.tumorous_images), len(self.nontumorous_images))
        self.tumorous_len = len(self.tumorous_images)
        self.nontumorous_len = len(self.nontumorous_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        tumorous_img = self.tumorous_images[index % self.tumorous_len]
        nontumorous_img = self.nontumorous_images[index % self.nontumorous_len]

        tumorous_path = os.path.join(self.root_tumorous, tumorous_img)
        nontumorous_path = os.path.join(self.root_nontumorous, nontumorous_img)

        tumorous_img = np.array(Image.open(tumorous_path).convert("RGB"))
        nontumorous_img = np.array(Image.open(nontumorous_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=tumorous_img, image0=nontumorous_img)
            tumorous_img = augmentations["image"]
            nontumorous_img = augmentations["image0"]

        return tumorous_img, nontumorous_img
