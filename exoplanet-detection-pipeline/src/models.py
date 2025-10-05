import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class TabularNet(nn.Module):
    """
    Multi-layer perceptron for tabular features.
    Handles variable input sizes with dropout and batch normalization.
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_sizes: List[int] = [256, 128, 64],
                 dropout_rate: float = 0.3,
                 use_batch_norm: bool = True,
                 output_size: int = 1):
        super(TabularNet, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        return self.network(x)

class ResidualCNN1D(nn.Module):
    """
    1D CNN for processing lightcurve residual windows.
    Uses residual connections and attention mechanisms.
    """
    
    def __init__(self,
                 input_length: int = 128,
                 n_filters: List[int] = [32, 64, 128, 256],
                 kernel_sizes: List[int] = [7, 5, 3, 3],
                 dropout_rate: float = 0.3,
                 output_size: int = 64):
        super(ResidualCNN1D, self).__init__()
        
        self.input_length = input_length
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes
        self.dropout_rate = dropout_rate
        
        # Initial convolution
        self.initial_conv = nn.Conv1d(1, n_filters[0], kernel_sizes[0], padding=kernel_sizes[0]//2)
        self.initial_bn = nn.BatchNorm1d(n_filters[0])
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for i in range(len(n_filters) - 1):
            self.residual_blocks.append(
                ResidualBlock1D(n_filters[i], n_filters[i + 1], kernel_sizes[i + 1])
            )
        
        # Attention mechanism
        self.attention = AttentionLayer1D(n_filters[-1])
        
        # Global pooling and output
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(n_filters[-1], output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, n_windows, sequence_length) or (batch_size, sequence_length)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Handle different input shapes
        if len(x.shape) == 2:
            # Shape: (batch_size, sequence_length) - single window per sample
            batch_size, seq_len = x.shape
            n_windows = 1
            x = x.unsqueeze(1)  # Add window dimension: (batch_size, 1, sequence_length)
        elif len(x.shape) == 3:
            # Shape: (batch_size, n_windows, sequence_length) - multiple windows per sample
            batch_size, n_windows, seq_len = x.shape
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.shape}")
        
        if n_windows == 0:
            # Handle case with no windows
            return torch.zeros(batch_size, self.fc.out_features, device=x.device)
        
        # Reshape for processing: (batch_size * n_windows, 1, seq_len)
        x = x.view(-1, 1, seq_len)
        
        # Initial convolution
        x = F.relu(self.initial_bn(self.initial_conv(x)))
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Attention
        x = self.attention(x)
        
        # Global pooling
        x = self.global_pool(x)  # (batch_size * n_windows, n_filters[-1], 1)
        x = x.squeeze(-1)        # (batch_size * n_windows, n_filters[-1])
        
        # Aggregate across windows (mean pooling)
        x = x.view(batch_size, n_windows, -1)  # (batch_size, n_windows, n_filters[-1])
        x = torch.mean(x, dim=1)               # (batch_size, n_filters[-1])
        
        # Output
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

class ResidualBlock1D(nn.Module):
    """Residual block for 1D CNN."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(ResidualBlock1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        
        x = x + residual
        x = F.relu(x)
        
        return x

class AttentionLayer1D(nn.Module):
    """Attention mechanism for 1D sequences."""
    
    def __init__(self, channels: int):
        super(AttentionLayer1D, self).__init__()
        
        self.channels = channels
        self.query = nn.Conv1d(channels, channels // 8, 1)
        self.key = nn.Conv1d(channels, channels // 8, 1)
        self.value = nn.Conv1d(channels, channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, length = x.shape
        
        q = self.query(x).view(batch_size, -1, length).permute(0, 2, 1)  # (B, L, C//8)
        k = self.key(x).view(batch_size, -1, length)                     # (B, C//8, L)
        v = self.value(x).view(batch_size, -1, length).permute(0, 2, 1)   # (B, L, C)
        
        # Attention weights
        attn = torch.softmax(torch.bmm(q, k) / (channels ** 0.5), dim=-1)  # (B, L, L)
        
        # Apply attention
        out = torch.bmm(attn, v).permute(0, 2, 1).view(batch_size, channels, length)
        
        return out + x  # Residual connection

class PixelCNN2D(nn.Module):
    """
    2D CNN for processing pixel difference images.
    Handles variable image sizes and phase dimensions.
    """
    
    def __init__(self,
                 n_phases: int = 32,
                 image_size: Tuple[int, int] = (10, 10),
                 n_filters: List[int] = [32, 64, 128],
                 dropout_rate: float = 0.3,
                 output_size: int = 64):
        super(PixelCNN2D, self).__init__()
        
        self.n_phases = n_phases
        self.image_size = image_size
        self.n_filters = n_filters
        self.dropout_rate = dropout_rate
        
        # Initial convolution - treat phases as channels
        self.initial_conv = nn.Conv2d(n_phases, n_filters[0], 3, padding=1)
        self.initial_bn = nn.BatchNorm2d(n_filters[0])
        
        # Convolutional blocks
        self.conv_blocks = nn.ModuleList()
        for i in range(len(n_filters) - 1):
            self.conv_blocks.append(
                ConvBlock2D(n_filters[i], n_filters[i + 1])
            )
        
        # Spatial attention
        self.spatial_attention = SpatialAttention()
        
        # Adaptive pooling to handle variable image sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Output layers
        final_size = n_filters[-1] * 4 * 4
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(final_size, 128)
        self.fc2 = nn.Linear(128, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, n_phases, height, width)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        if x.numel() == 0:
            # Handle empty input
            return torch.zeros(x.shape[0], self.fc2.out_features, device=x.device)
        
        # Initial convolution
        x = F.relu(self.initial_bn(self.initial_conv(x)))
        
        # Convolutional blocks
        for block in self.conv_blocks:
            x = block(x)
        
        # Spatial attention
        x = self.spatial_attention(x)
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten and output
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class ConvBlock2D(nn.Module):
    """Convolutional block for 2D CNN."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock2D, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.dropout = nn.Dropout2d(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        return x

class SpatialAttention(nn.Module):
    """Spatial attention mechanism for 2D feature maps."""
    
    def __init__(self):
        super(SpatialAttention, self).__init__()
        
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        
        concat = torch.cat([avg_pool, max_pool], dim=1)
        attention = torch.sigmoid(self.conv(concat))
        
        return x * attention

class HybridEnsemble(nn.Module):
    """
    Hybrid ensemble combining tabular, 1D CNN, and 2D CNN models.
    Uses late fusion with learnable weights.
    """
    
    def __init__(self,
                 tabular_size: int,
                 cnn1d_output_size: int = 64,
                 cnn2d_output_size: int = 64,
                 fusion_hidden_size: int = 32,
                 dropout_rate: float = 0.3):
        super(HybridEnsemble, self).__init__()
        
        # Individual models
        self.tabular_net = TabularNet(
            input_size=tabular_size,
            hidden_sizes=[256, 128, 64],
            output_size=cnn1d_output_size
        )
        
        self.cnn1d = ResidualCNN1D(output_size=cnn1d_output_size)
        self.cnn2d = PixelCNN2D(output_size=cnn2d_output_size)
        
        # Fusion layers
        fusion_input_size = cnn1d_output_size + cnn1d_output_size + cnn2d_output_size
        self.fusion_fc1 = nn.Linear(fusion_input_size, fusion_hidden_size)
        self.fusion_bn = nn.BatchNorm1d(fusion_hidden_size)
        self.fusion_dropout = nn.Dropout(dropout_rate)
        self.fusion_fc2 = nn.Linear(fusion_hidden_size, 1)
        
        # Attention weights for fusion
        self.attention_weights = nn.Parameter(torch.ones(3) / 3)
        
    def forward(self, 
                tabular_data: torch.Tensor,
                residual_windows: torch.Tensor,
                pixel_differences: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the hybrid ensemble.
        
        Args:
            tabular_data: Tabular features (batch_size, n_features)
            residual_windows: 1D residual windows (batch_size, n_windows, seq_len)
            pixel_differences: 2D pixel differences (batch_size, n_phases, height, width)
            
        Returns:
            Tuple of (final_output, individual_outputs)
        """
        # Individual model outputs
        tabular_out = self.tabular_net(tabular_data)
        cnn1d_out = self.cnn1d(residual_windows)
        cnn2d_out = self.cnn2d(pixel_differences)
        
        # Apply attention weights
        weighted_outputs = [
            self.attention_weights[0] * tabular_out,
            self.attention_weights[1] * cnn1d_out,
            self.attention_weights[2] * cnn2d_out
        ]
        
        # Concatenate for fusion
        fusion_input = torch.cat(weighted_outputs, dim=1)
        
        # Fusion layers
        x = F.relu(self.fusion_bn(self.fusion_fc1(fusion_input)))
        x = self.fusion_dropout(x)
        final_output = self.fusion_fc2(x)
        
        # Store individual outputs for analysis
        individual_outputs = {
            'tabular': tabular_out,
            'cnn1d': cnn1d_out,
            'cnn2d': cnn2d_out,
            'attention_weights': self.attention_weights
        }
        
        return final_output, individual_outputs

def create_models(config: Dict, model_types: List[str] = None) -> Dict[str, nn.Module]:
    """
    Create all model architectures based on configuration.
    
    Args:
        config: Configuration dictionary with model parameters
        model_types: List of model types to create (default: all)
        
    Returns:
        Dictionary containing requested models
    """
    if model_types is None:
        model_types = ['tabular', 'cnn1d', 'cnn2d', 'hybrid']
    
    models = {}
    
    # TabularNet
    if 'tabular' in model_types:
        models['tabular'] = TabularNet(
            input_size=config.get('tabular_input_size', 50),
            hidden_sizes=config.get('tabular_hidden_sizes', [256, 128, 64]),
            dropout_rate=config.get('dropout_rate', 0.3),
            output_size=1
        )
    
        # ResidualCNN1D  
    if 'cnn1d' in model_types:
        models['cnn1d'] = ResidualCNN1D(
            input_length=config.get('cnn1d_input_length', 128),
            n_filters=config.get('cnn1d_filters', [64, 128, 256]),
            dropout_rate=config.get('dropout_rate', 0.3),
            output_size=1
        )
    
    # PixelCNN2D
    if 'cnn2d' in model_types:
        models['cnn2d'] = PixelCNN2D(
            n_phases=config.get('cnn2d_phases', 32),
            image_size=config.get('cnn2d_image_size', (10, 10)),
            n_filters=config.get('cnn2d_filters', [32, 64, 128]),
            dropout_rate=config.get('dropout_rate', 0.3),
            output_size=1
        )
    
    # Hybrid Ensemble
    if 'hybrid' in model_types:
        models['hybrid'] = HybridEnsemble(
            tabular_size=config.get('tabular_input_size', 50),
            cnn1d_output_size=config.get('fusion_feature_size', 64),
            cnn2d_output_size=config.get('fusion_feature_size', 64),
            fusion_hidden_size=config.get('fusion_hidden_size', 32),
            dropout_rate=config.get('dropout_rate', 0.3)
        )
    
    logger.info(f"Created {len(models)} models: {list(models.keys())}")
    
    return models

def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(models: Dict[str, nn.Module]):
    """Print summary of all models."""
    print("\n" + "="*50)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*50)
    
    for name, model in models.items():
        n_params = count_parameters(model)
        print(f"\n{name.upper()}:")
        print(f"  Parameters: {n_params:,}")
        print(f"  Architecture: {model.__class__.__name__}")
    
    total_params = sum(count_parameters(model) for model in models.values())
    print(f"\nTOTAL PARAMETERS: {total_params:,}")
    print("="*50)

if __name__ == "__main__":
    # Test model creation
    config = {
        'tabular_input_size': 50,
        'cnn1d_input_length': 128,
        'cnn2d_phases': 32,
        'cnn2d_image_size': (10, 10),
        'dropout_rate': 0.3
    }
    
    models = create_models(config)
    print_model_summary(models)
    
    # Test forward passes
    batch_size = 4
    
    # Create dummy data
    tabular_data = torch.randn(batch_size, 50)
    residual_windows = torch.randn(batch_size, 5, 128)  # 5 windows per sample
    pixel_differences = torch.randn(batch_size, 32, 10, 10)
    
    print("\nTesting forward passes...")
    
    # Test individual models
    with torch.no_grad():
        tabular_out = models['tabular'](tabular_data)
        cnn1d_out = models['cnn1d'](residual_windows)
        cnn2d_out = models['cnn2d'](pixel_differences)
        hybrid_out, individual_outs = models['hybrid'](tabular_data, residual_windows, pixel_differences)
    
    print(f"Tabular output shape: {tabular_out.shape}")
    print(f"CNN1D output shape: {cnn1d_out.shape}")
    print(f"CNN2D output shape: {cnn2d_out.shape}")
    print(f"Hybrid output shape: {hybrid_out.shape}")
    
    print("\nModel tests passed!")