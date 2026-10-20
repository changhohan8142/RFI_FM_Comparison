import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as tfms
from PIL import Image


class DementiaDetectionDataset(Dataset):
    """Development dataset for prevalent dementia detection (one row per fundus image).

    csv_path: dementia_detection.csv from data_preparation.ipynb (columns pngfilename, label, train, valid, test)
    kind    : 'train', 'valid' or 'test'
    Returns (image_tensor, label).
    """
    def __init__(self, csv_path, kind, img_sz):
        df = pd.read_csv(csv_path, low_memory=False)
        self.df = df[df[kind]==True][['pngfilename', 'label']]
        norm = tfms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if kind == 'train':
            self.tfms = tfms.Compose([
                tfms.RandomResizedCrop((img_sz, img_sz)),
                tfms.AutoAugment(),
                tfms.RandomHorizontalFlip(),
                tfms.ToTensor(),
                norm,
            ])
        else:
            self.tfms = tfms.Compose([
                tfms.Resize((img_sz, img_sz)),
                tfms.ToTensor(),
                norm
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = self.tfms(
            Image.open(r.pngfilename).convert("RGB")
        )
        label = torch.tensor(r.label, dtype=torch.float32)
        return img, label


class DementiaPredictionDataset(Dataset):
    """Future-incidence cohort (one row per fundus image); the detection models are applied without further training.

    csv_path: dementia_prediction.csv from data_preparation.ipynb (columns pngfilename, event, obs_time)
    Returns (image_tensor, event, obs_time).
    """
    def __init__(self, csv_path, img_sz):
        df = pd.read_csv(csv_path, low_memory=False)
        self.df = df[['pngfilename', 'event', 'obs_time']]

        self.tfms = tfms.Compose([
            tfms.Resize((img_sz, img_sz)),
            tfms.ToTensor(),
            tfms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = self.tfms(
            Image.open(r.pngfilename).convert("RGB")
        )
        event = torch.tensor(r.event, dtype=torch.float32)
        dur = torch.tensor(r.obs_time, dtype=torch.int32)
        return img, event, dur
