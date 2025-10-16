import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SimpleImageDataset(Dataset):
    def __init__(self, image_folder, image_size=64):
        self.image_paths = [
            os.path.join(image_folder, f) 
            for f in os.listdir(image_folder) 
            if f.endswith(('png', 'jpg', 'jpeg'))
        ]
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize ke [-1, 1]
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(image)
