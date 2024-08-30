import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import torchvision.transforms as transforms
from PIL import Image

class PokemonDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.pokemon_metadata = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.pokemon_metadata)

    def __getitem__(self, idx):
        row = self.pokemon_metadata.iloc[idx]
        img_path = row['Path']
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # return {
        #     'image': image,
        #     'name': name,
        #     'type1': row['Type1'],
        #     'type2': row['Type2'],
        #     'evolution': row['Evolution']
        # }
        return image
