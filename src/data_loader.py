import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

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
        
        image = self.load_image(img_path)
        label = self.convert_label(row)
        patient_id = row['PatientID']

        if self.transform:
            image = self.transform(image)

        return image, label, patient_id

    def convert_label(self, row):
        if row['Normal'] == 1:
            return 0
        elif row['Actionable'] == 1:
            return 1
        elif row['Benign'] == 1:
            return 2
        elif row['Cancer'] == 1:
            return 3
        else:
            return -1

    def load_image(self, img_path):
        try:
            image = Image.open(img_path).convert('RGB')
            return image
        except:
            return Image.fromarray((torch.zeros((256, 256, 3)).numpy().astype('uint8')))

