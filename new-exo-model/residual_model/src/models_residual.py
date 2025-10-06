import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=7, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.dropout(h)
        h = self.bn2(self.conv2(h))
        return F.relu(h + residual)

class ResidualCNN1D(nn.Module):
    """
    CNN for variable-length residual sequences.
    Input: [B, features, seq_len] where features=128
    Output: logits for binary classification
    """
    def __init__(self, in_features=128, base_channels=64, max_length=512):
        super().__init__()
        
        # Feature projection: map 128 features to manageable number of channels
        self.feature_proj = nn.Sequential(
            nn.Conv1d(in_features, base_channels, kernel_size=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU()
        )
        
        # Stem: initial conv with larger kernel
        self.stem = nn.Sequential(
            nn.Conv1d(base_channels, base_channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
        )
        
        # Residual blocks
        self.block1 = ResidualBlock1D(base_channels, kernel_size=9, dropout=0.1)
        self.block2 = ResidualBlock1D(base_channels, kernel_size=7, dropout=0.1)
        self.block3 = ResidualBlock1D(base_channels, kernel_size=5, dropout=0.1)
        
        # Wide temporal context
        self.wide_conv = nn.Sequential(
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=21, padding=10),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        )
        
        # Another set of residual blocks with more channels
        self.block4 = ResidualBlock1D(base_channels * 2, kernel_size=7, dropout=0.15)
        self.block5 = ResidualBlock1D(base_channels * 2, kernel_size=5, dropout=0.15)
        
        # Global pooling and classification head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(base_channels * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: [B, features=128, seq_len]
        
        # Project features to manageable number of channels
        h = self.feature_proj(x)  # [B, base_channels, seq_len]
        
        # Stem processing
        h = self.stem(h)  # [B, base_channels, seq_len//2]
        
        # First set of residual blocks
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        
        # Wide temporal context
        h = self.wide_conv(h)  # [B, base_channels*2, seq_len//4]
        
        # Second set of residual blocks
        h = self.block4(h)
        h = self.block5(h)
        
        # Global average pooling
        h = self.global_pool(h).squeeze(-1)  # [B, base_channels*2]
        
        # Classification head
        logits = self.head(h).squeeze(-1)  # [B]
        
        return logits


class LightweightResidualCNN(nn.Module):
    """
    Lighter version for faster training/inference
    """
    def __init__(self, in_features=128, base_channels=32, max_length=512):
        super().__init__()
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Conv1d(in_features, base_channels, kernel_size=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU()
        )
        
        # Multi-scale convolutions
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(base_channels, base_channels, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(base_channels),
                nn.ReLU(),
                nn.MaxPool1d(2)
            ) for k in [11, 7, 5]
        ])
        
        # Final processing
        self.final_conv = nn.Sequential(
            nn.Conv1d(base_channels * 3, base_channels * 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Classification head
        self.head = nn.Sequential(
            nn.Linear(base_channels * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        # x: [B, features=128, seq_len]
        h = self.feature_proj(x)
        
        # Multi-scale processing
        features = []
        for conv_layer in self.conv_layers:
            features.append(conv_layer(h))
        
        # Concatenate multi-scale features
        h = torch.cat(features, dim=1)  # [B, base_channels*3, seq_len//2]
        
        # Final processing
        h = self.final_conv(h).squeeze(-1)  # [B, base_channels*2]
        
        # Classification
        logits = self.head(h).squeeze(-1)  # [B]
        
        return logits