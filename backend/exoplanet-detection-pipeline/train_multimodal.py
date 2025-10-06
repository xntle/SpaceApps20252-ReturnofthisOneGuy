"""
Multi-modal training pipeline for exoplanet detection
Combines tabular features, 1D CNN (residual windows), and 2D CNN (pixel differences)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
from typing import Dict, Tuple
import os

from src.data_loader import load_and_prepare_data, create_train_val_test_splits
from src.cnn_data_loader import load_cnn_data, create_cnn_datasets
from src.models import TabularNet, ResidualCNN1D, PixelCNN2D

class MultiModalDataset(Dataset):
    """Dataset for multi-modal training"""
    
    def __init__(self, tabular_X, cnn1d_X, cnn2d_X, y):
        self.tabular_X = torch.FloatTensor(tabular_X)
        self.cnn1d_X = torch.FloatTensor(cnn1d_X)
        self.cnn2d_X = torch.FloatTensor(cnn2d_X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return {
            'tabular': self.tabular_X[idx],
            'cnn1d': self.cnn1d_X[idx],
            'cnn2d': self.cnn2d_X[idx],
            'label': self.y[idx]
        }

class MultiModalFusionModel(nn.Module):
    """Fusion model combining tabular + CNN1D + CNN2D"""
    
    def __init__(self, n_tabular_features=39):
        super().__init__()
        
        # Individual models
        self.tabular_model = TabularNet(input_size=n_tabular_features)
        self.cnn1d_model = ResidualCNN1D()
        self.cnn2d_model = PixelCNN2D()
        
        # Fusion layers - tabular (1) + cnn1d (64) + cnn2d (64) = 129 features
        self.fusion = nn.Sequential(
            nn.Linear(129, 64),  # 1 + 64 + 64 = 129 inputs
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, tabular_x, cnn1d_x, cnn2d_x):
        # Get predictions from each model
        tabular_out = self.tabular_model(tabular_x)
        cnn1d_out = self.cnn1d_model(cnn1d_x)
        cnn2d_out = self.cnn2d_model(cnn2d_x)
        
        # Concatenate outputs
        combined = torch.cat([tabular_out, cnn1d_out, cnn2d_out], dim=1)
        
        # Final fusion
        output = self.fusion(combined)
        
        return output

def train_multimodal_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 1e-3
) -> Dict:
    """Train the multi-modal fusion model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_auc = 0
    best_model_state = None
    train_losses = []
    val_aucs = []
    
    print("🚀 Starting multi-modal training...")
    print(f"📱 Device: {device}")
    print(f"📊 Training samples: {len(train_loader.dataset)}")
    print(f"📊 Validation samples: {len(val_loader.dataset)}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            tabular = batch['tabular'].to(device)
            cnn1d = batch['cnn1d'].to(device)
            cnn2d = batch['cnn2d'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(tabular, cnn1d, cnn2d)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                tabular = batch['tabular'].to(device)
                cnn1d = batch['cnn1d'].to(device)
                cnn2d = batch['cnn2d'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(tabular, cnn1d, cnn2d)
                
                val_preds.extend(outputs.cpu().numpy().flatten())
                val_labels.extend(labels.cpu().numpy())
        
        val_auc = roc_auc_score(val_labels, val_preds)
        train_loss = train_loss / len(train_loader)
        
        train_losses.append(train_loss)
        val_aucs.append(val_auc)
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
        
        # Learning rate scheduling
        scheduler.step(1 - val_auc)  # Minimize (1 - AUC)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Loss={train_loss:.4f}, Val AUC={val_auc:.4f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    print(f"✅ Training complete! Best validation AUC: {best_val_auc:.4f}")
    
    return {
        'model': model,
        'best_val_auc': best_val_auc,
        'train_losses': train_losses,
        'val_aucs': val_aucs
    }

def evaluate_multimodal_model(model: nn.Module, test_loader: DataLoader) -> Dict:
    """Evaluate the multi-modal model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            tabular = batch['tabular'].to(device)
            cnn1d = batch['cnn1d'].to(device)
            cnn2d = batch['cnn2d'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(tabular, cnn1d, cnn2d)
            
            test_preds.extend(outputs.cpu().numpy().flatten())
            test_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    test_auc = roc_auc_score(test_labels, test_preds)
    test_preds_binary = (np.array(test_preds) > 0.5).astype(int)
    test_accuracy = accuracy_score(test_labels, test_preds_binary)
    
    return {
        'auc': test_auc,
        'accuracy': test_accuracy,
        'predictions': test_preds,
        'labels': test_labels,
        'predictions_binary': test_preds_binary
    }

def main():
    """Main training pipeline"""
    
    print("🌟 Multi-Modal Exoplanet Detection Training Pipeline")
    print("=" * 60)
    
    # Load tabular data
    data = load_and_prepare_data()
    splits = create_train_val_test_splits(data)
    feature_names = splits['feature_names']
    print(f"📊 Loaded tabular data: {len(feature_names)} features")
    
    # Load CNN data
    cnn_data = load_cnn_data()
    
    # Create aligned CNN datasets
    cnn_datasets = create_cnn_datasets(cnn_data, splits, feature_names)
    
    # Create data loaders - use only samples that have CNN data
    # Get the minimum number of samples across all modalities
    min_train_samples = min(len(splits['train']['y']), len(cnn_datasets['train']['y']))
    min_val_samples = min(len(splits['val']['y']), len(cnn_datasets['val']['y']))
    min_test_samples = min(len(splits['test']['y']), len(cnn_datasets['test']['y']))
    
    print(f"🔗 Aligning datasets to CNN sample sizes:")
    print(f"   Train: {min_train_samples} samples")
    print(f"   Val: {min_val_samples} samples") 
    print(f"   Test: {min_test_samples} samples")
    
    train_dataset = MultiModalDataset(
        splits['train']['X'][:min_train_samples], 
        cnn_datasets['train']['cnn1d'][:min_train_samples], 
        cnn_datasets['train']['cnn2d'][:min_train_samples], 
        splits['train']['y'][:min_train_samples]
    )
    
    val_dataset = MultiModalDataset(
        splits['val']['X'][:min_val_samples], 
        cnn_datasets['val']['cnn1d'][:min_val_samples], 
        cnn_datasets['val']['cnn2d'][:min_val_samples], 
        splits['val']['y'][:min_val_samples]
    )
    
    test_dataset = MultiModalDataset(
        splits['test']['X'][:min_test_samples], 
        cnn_datasets['test']['cnn1d'][:min_test_samples], 
        cnn_datasets['test']['cnn2d'][:min_test_samples], 
        splits['test']['y'][:min_test_samples]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create and train model
    model = MultiModalFusionModel(n_tabular_features=len(feature_names))
    
    training_results = train_multimodal_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=100,
        learning_rate=1e-3
    )
    
    # Evaluate on test set
    test_results = evaluate_multimodal_model(training_results['model'], test_loader)
    
    print("\n🎯 FINAL RESULTS")
    print("=" * 40)
    print(f"🏆 Test AUC: {test_results['auc']:.4f}")
    print(f"🎯 Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"📈 Best Val AUC: {training_results['best_val_auc']:.4f}")
    
    # Save model
    model_path = "models/multimodal_fusion_model.pth"
    torch.save(training_results['model'].state_dict(), model_path)
    print(f"💾 Model saved to {model_path}")
    
    return training_results, test_results

if __name__ == "__main__":
    main()