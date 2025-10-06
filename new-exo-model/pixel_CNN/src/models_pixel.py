#!/usr/bin/env python3
"""
Compact 2D CNN for pixel difference classification
"""
import torch
import torch.nn as nn

class PixelCNN2D(nn.Module):
    """Compact CNN for pixel difference images with adaptive pooling"""
    
    def __init__(self, base=16, dropout=0.3):
        """
        Args:
            base: Base number of channels (scales: base, 2*base, 4*base)
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        # Convolutional backbone with adaptive pooling
        self.backbone = nn.Sequential(
            # First conv block
            nn.Conv2d(1, base, kernel_size=3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Second conv block  
            nn.Conv2d(base, base*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Third conv block
            nn.Conv2d(base*2, base*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base*4),
            nn.ReLU(inplace=True),
            
            # Adaptive pooling to handle variable image sizes
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # Classification head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base*4, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        """Forward pass"""
        h = self.backbone(x)
        return self.head(h).squeeze(-1)  # Return logits

    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class LightweightPixelCNN(nn.Module):
    """Ultra-lightweight CNN for very small datasets"""
    
    def __init__(self, base=8, dropout=0.4):
        super().__init__()
        
        self.backbone = nn.Sequential(
            # Single conv block with heavy dropout
            nn.Conv2d(1, base, kernel_size=5, padding=2),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4),
            nn.Dropout2d(dropout/2),
            
            nn.Conv2d(base, base*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base*2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base*2, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        h = self.backbone(x)
        return self.head(h).squeeze(-1)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def get_model(model_type="standard", base=16, dropout=0.3):
    """Factory function to create models"""
    if model_type == "lightweight":
        model = LightweightPixelCNN(base=base//2, dropout=dropout)
    else:
        model = PixelCNN2D(base=base, dropout=dropout)
    
    print(f"Created {model_type} model with {model.count_parameters():,} parameters")
    return model