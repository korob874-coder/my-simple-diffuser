import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        # Encoder
        self.enc1 = self._block(in_channels, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        
        # Decoder  
        self.dec2 = self._block(256 + 128, 128)  # Skip connection
        self.dec1 = self._block(128 + 64, 64)    # Skip connection
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )
    
    def forward(self, x, t):
        # Encoder
        e1 = self.enc1(x)                    # [batch, 64, 64, 64]
        e2 = self.enc2(F.max_pool2d(e1, 2))  # [batch, 128, 32, 32] 
        e3 = self.enc3(F.max_pool2d(e2, 2))  # [batch, 256, 16, 16]
        
        # Decoder dengan skip connections + UPSAMPLING
        # Upsample e3 untuk match ukuran e2
        e3_upsampled = F.interpolate(e3, scale_factor=2, mode='bilinear', align_corners=False)
        # e3_upsampled: [batch, 256, 32, 32]
        # e2: [batch, 128, 32, 32] 
        
        d2 = self.dec2(torch.cat([e3_upsampled, e2], dim=1))  # [batch, 256+128=384, 32, 32] -> [batch, 128, 32, 32]
        
        # Upsample d2 untuk match ukuran e1
        d2_upsampled = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        # d2_upsampled: [batch, 128, 64, 64]
        # e1: [batch, 64, 64, 64]
        
        d1 = self.dec1(torch.cat([d2_upsampled, e1], dim=1))  # [batch, 128+64=192, 64, 64] -> [batch, 64, 64, 64]
        
        return self.final(d1)
