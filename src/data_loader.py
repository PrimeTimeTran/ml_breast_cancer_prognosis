import os
import torch
import struct
import pandas as pd
import numpy as np
from skimage import io
from array import array
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class DataLoader(Dataset):
    def __init__(self, csv_file, set_type):
        self.data = pd.read_csv(csv_file)
        self.set_type = set_type
        self.image_dir = f'tmp/imgs/{set_type}'
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, f"{row['PatientID']}-{row['View']}.png")
        image = io.imread(img_path)
        label = self.convert_label(row)

        if self.transform:
            image = self.transform(image)

        return image, label
    def convert_label(self, row):
        if row['Normal'] == 1:
            return 0
        elif row['Actionable'] == 1:
            return 1
        elif row['Benign'] == 1:
            return 2
        elif row['Cancer'] == 1:
            return 3

    def load_image(self, img_path):
        try:
            image = plt.imread(img_path)
            return image
        except:
            return torch.zeros((3, 256, 256))  # Return a dummy image tensor
