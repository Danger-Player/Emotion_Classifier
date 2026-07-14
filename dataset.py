import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

def loadData(csv_path):
    df = pd.read_csv(csv_path)
    train_df = df[df['Usage'] == 'Training'].reset_index(drop=True)
    val_df = df[df['Usage'] != 'Training'].reset_index(drop=True)
    return train_df, val_df

class FERDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        data = []
        for pixels_str in dataframe['pixels']:
            pixels_list = pixels_str.split()
            pixels_array = np.array(pixels_list, dtype=np.uint8)
            pixels_reshaped = pixels_array.reshape(48, 48)
            data.append(pixels_reshaped)

        self.images = np.stack(data)
        self.labels = dataframe['emotion'].values
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = image[:,:,np.newaxis]
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label
    
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
])

def getDatasets(csv_path):
    train_df, val_df = loadData(csv_path)
    train_dataset = FERDataset(train_df, transform=train_transform)
    val_dataset = FERDataset(val_df, transform=val_transform)
    return train_dataset, val_dataset
