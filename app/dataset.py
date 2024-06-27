import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = 'D:/multitask_prediction'

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(os.path.join(PROJECT_ROOT, csv_file))
        self.img_dir = os.path.join(PROJECT_ROOT, img_dir)
        self.transform = transform
        
        self.nationality_encoder = LabelEncoder()
        self.emotion_encoder = LabelEncoder()
        self.dress_color_encoder = LabelEncoder()
        
        self.annotations['nationality'] = self.nationality_encoder.fit_transform(self.annotations['nationality'])
        self.annotations['emotion'] = self.emotion_encoder.fit_transform(self.annotations['emotion'])
        self.annotations['dress_color'] = self.dress_color_encoder.fit_transform(self.annotations['dress_color'])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        nationality = self.annotations.iloc[idx, 1]
        emotion = self.annotations.iloc[idx, 2]
        age = self.annotations.iloc[idx, 3]
        dress_color = self.annotations.iloc[idx, 4]

        if self.transform:
            image = self.transform(image)

        return image, nationality, emotion, age, dress_color