#!/usr/bin/env python3
"""Test model loading independently"""

import torch
import torch.nn as nn

# Residual model architecture matching saved weights
class LightweightResidualCNN(nn.Module):
    def __init__(self, in_features=128, base_channels=16):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Conv1d(in_features, base_channels, kernel_size=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels, kernel_size=11, padding=5),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.head = nn.Sequential(
            nn.Linear(base_channels * 2, base_channels * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(base_channels * 2, 1)
        )
    
    def forward(self, x):
        h = self.feature_proj(x)
        h = self.conv1(h)
        h = self.conv2(h).squeeze(-1)
        return self.head(h).squeeze(-1)

# Pixel model architecture matching saved weights  
class PixelCNN2D(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, base, kernel_size=3, padding=1),     # 0
            nn.BatchNorm2d(base),                             # 1
            nn.ReLU(inplace=True),                            # 2
            nn.MaxPool2d(2),                                  # 3
            nn.Conv2d(base, base*2, kernel_size=3, padding=1), # 4
            nn.BatchNorm2d(base*2),                           # 5
            nn.ReLU(inplace=True),                            # 6
            nn.MaxPool2d(2),                                  # 7
            nn.Conv2d(base*2, base*4, kernel_size=3, padding=1), # 8
            nn.BatchNorm2d(base*4),                           # 9
            nn.ReLU(inplace=True),                            # 10
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 0 - Global average pooling to get 64x1x1
            nn.Flatten(),                  # 1 - Flatten to 64
            nn.Linear(base*4, base*2),     # 2 - Linear(64, 32)
            nn.ReLU(inplace=True),         # 3
            nn.Dropout(0.3),               # 4
            nn.Linear(base*2, 1)           # 5 - Linear(32, 1)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        return self.head(x).squeeze(-1)

# Test residual model
print("Testing Residual CNN...")
res_model = LightweightResidualCNN()
res_state = torch.load('/home/roshan/Desktop/new-exo-model/residual_model/models/residual_cnn_best_fold0.pt', map_location='cpu')
res_model.load_state_dict(res_state)
res_model.eval()  # Important: set to eval mode to avoid BatchNorm issues
print("✅ Residual CNN loaded successfully!")

# Test with correct input shape
x_res = torch.randn(1, 128, 2)  # [batch, features, time]
with torch.no_grad():
    out_res = torch.sigmoid(res_model(x_res))
    print(f"✅ Residual prediction: {out_res.item():.4f}")

# Test pixel model  
print("\nTesting Pixel CNN...")
pix_model = PixelCNN2D()
pix_state = torch.load('/home/roshan/Desktop/new-exo-model/pixel_CNN/models/pixel_cnn_best_fold0.pt', map_location='cpu')
pix_model.load_state_dict(pix_state)
pix_model.eval()  # Important: set to eval mode
print("✅ Pixel CNN loaded successfully!")

# Test with correct input shape
x_pix = torch.randn(1, 1, 24, 24)  # [batch, channels, height, width]
with torch.no_grad():
    out_pix = torch.sigmoid(pix_model(x_pix))
    print(f"✅ Pixel prediction: {out_pix.item():.4f}")

print("\n🎉 All models loaded and tested successfully!")