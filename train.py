import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import SimpleUNet
from dataset import SimpleImageDataset
import os
from tqdm import tqdm

class SimpleDiffuser:
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        self.beta = torch.linspace(1e-4, 0.02, timesteps)
        self.alpha = 1. - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
    
    def add_noise(self, x_0, t):
        """Tambahkan noise ke gambar sesuai timestep"""
        sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1. - self.alpha_cumprod[t]).view(-1, 1, 1, 1)
        
        noise = torch.randn_like(x_0)
        x_t = sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise
        return x_t, noise
    
    def train(self, model, dataloader, epochs=100, device='cuda'):
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}'):
                batch = batch.to(device)
                
                # Random timestep
                t = torch.randint(0, self.timesteps, (batch.size(0),), device=device)
                
                # Add noise
                noisy_imgs, noise = self.add_noise(batch, t)
                
                # Predict noise
                pred_noise = model(noisy_imgs, t)
                
                # Loss
                loss = criterion(pred_noise, noise)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}')
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')

if __name__ == "__main__":
    # Config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_size = 64
    batch_size = 16
    epochs = 50
    
    # Dataset
    dataset = SimpleImageDataset('data/your_images', image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model & Trainer
    model = SimpleUNet()
    diffuser = SimpleDiffuser()
    
    # Train
    diffuser.train(model, dataloader, epochs=epochs, device=device)
