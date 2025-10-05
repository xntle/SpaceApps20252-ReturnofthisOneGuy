"""
Enhanced Multi-Modal Training with Scaling Support
==================================================

Supports larger CNN datasets and provides detailed performance tracking
to demonstrate the value of increased CNN coverage.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from typing import Dict, Tuple
import os
import time
from pathlib import Path

from src.data_loader import load_and_prepare_data, create_train_val_test_splits
from src.cnn_data_loader import load_cnn_data, create_cnn_datasets
from src.models import TabularNet, ResidualCNN1D, PixelCNN2D
from src.threshold_optimization import threshold_for_fpr, validate_threshold_performance

class MultiModalDataset(Dataset):
    """Enhanced dataset for multi-modal training with better memory handling"""
    
    def __init__(self, tabular_X, cnn1d_X, cnn2d_X, y, device='cpu'):
        # Convert to tensors and move to device
        self.tabular_X = torch.FloatTensor(tabular_X).to(device)
        self.cnn1d_X = torch.FloatTensor(cnn1d_X).to(device)
        self.cnn2d_X = torch.FloatTensor(cnn2d_X).to(device)
        self.y = torch.FloatTensor(y).to(device)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return {
            'tabular': self.tabular_X[idx],
            'cnn1d': self.cnn1d_X[idx],
            'cnn2d': self.cnn2d_X[idx],
            'label': self.y[idx]
        }

class EnhancedMultiModalFusionModel(nn.Module):
    """Enhanced fusion model with better architecture and dropout"""
    
    def __init__(self, n_tabular_features=39, fusion_dropout=0.3):
        super().__init__()
        
        # Individual models with frozen option
        self.tabular_model = TabularNet(input_size=n_tabular_features)
        self.cnn1d_model = ResidualCNN1D()
        self.cnn2d_model = PixelCNN2D()
        
        # Enhanced fusion network
        self.fusion = nn.Sequential(
            nn.Linear(129, 128),  # tabular(1) + cnn1d(64) + cnn2d(64)
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(fusion_dropout),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(fusion_dropout * 0.7),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(fusion_dropout * 0.5),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, tabular_x, cnn1d_x, cnn2d_x):
        # Get outputs from individual models
        tabular_out = self.tabular_model(tabular_x)
        cnn1d_out = self.cnn1d_model(cnn1d_x)
        cnn2d_out = self.cnn2d_model(cnn2d_x)
        
        # Concatenate all outputs
        combined = torch.cat([tabular_out, cnn1d_out, cnn2d_out], dim=1)
        
        # Final fusion prediction
        output = self.fusion(combined)
        
        return output
    
    def get_individual_predictions(self, tabular_x, cnn1d_x, cnn2d_x):
        """Get predictions from individual models for analysis"""
        with torch.no_grad():
            tabular_out = self.tabular_model(tabular_x)
            cnn1d_out = self.cnn1d_model(cnn1d_x)
            cnn2d_out = self.cnn2d_model(cnn2d_x)
            
            return {
                'tabular_features': tabular_out.cpu().numpy(),
                'cnn1d_features': cnn1d_out.cpu().numpy(),
                'cnn2d_features': cnn2d_out.cpu().numpy()
            }

def train_enhanced_multimodal_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    patience: int = 15,
    min_delta: float = 0.001
) -> Dict:
    """Enhanced training with early stopping and better logging"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=patience//2, factor=0.5
    )
    
    best_val_auc = 0
    best_model_state = None
    patience_counter = 0
    train_losses = []
    val_aucs = []
    val_accuracies = []
    
    print("🚀 Starting enhanced multi-modal training...")
    print(f"📱 Device: {device}")
    print(f"📊 Training samples: {len(train_loader.dataset)}")
    print(f"📊 Validation samples: {len(val_loader.dataset)}")
    print(f"⏱️  Early stopping patience: {patience} epochs")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            tabular = batch['tabular']
            cnn1d = batch['cnn1d']
            cnn2d = batch['cnn2d']
            labels = batch['label'].unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(tabular, cnn1d, cnn2d)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        # Validation phase
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                tabular = batch['tabular']
                cnn1d = batch['cnn1d']
                cnn2d = batch['cnn2d']
                labels = batch['label']
                
                outputs = model(tabular, cnn1d, cnn2d)
                
                val_preds.extend(outputs.cpu().numpy().flatten())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        val_auc = roc_auc_score(val_labels, val_preds)
        val_acc = accuracy_score(val_labels, (np.array(val_preds) > 0.5).astype(int))
        train_loss = train_loss / num_batches
        
        train_losses.append(train_loss)
        val_aucs.append(val_auc)
        val_accuracies.append(val_acc)
        
        # Early stopping check
        if val_auc > best_val_auc + min_delta:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Learning rate scheduling
        scheduler.step(val_auc)
        
        # Progress logging
        if epoch % 10 == 0 or patience_counter >= patience:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:3d}: Loss={train_loss:.4f}, Val AUC={val_auc:.4f}, "
                  f"Val Acc={val_acc:.4f}, Time={elapsed:.1f}s")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"🛑 Early stopping at epoch {epoch} (patience={patience})")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    total_time = time.time() - start_time
    print(f"✅ Training complete! Best validation AUC: {best_val_auc:.4f}")
    print(f"⏱️  Total training time: {total_time:.1f}s")
    
    return {
        'model': model,
        'best_val_auc': best_val_auc,
        'train_losses': train_losses,
        'val_aucs': val_aucs,
        'val_accuracies': val_accuracies,
        'training_time': total_time
    }

def evaluate_with_threshold_optimization(
    model: nn.Module, 
    test_loader: DataLoader,
    val_loader: DataLoader = None,
    target_fpr: float = 0.05
) -> Dict:
    """Evaluate model with optimized threshold for target FPR"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Get validation predictions for threshold optimization
    if val_loader is not None:
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                tabular = batch['tabular']
                cnn1d = batch['cnn1d']
                cnn2d = batch['cnn2d']
                labels = batch['label']
                
                outputs = model(tabular, cnn1d, cnn2d)
                
                val_preds.extend(outputs.cpu().numpy().flatten())
                val_labels.extend(labels.cpu().numpy())
        
        # Optimize threshold on validation set
        optimal_threshold = threshold_for_fpr(val_labels, val_preds, target_fpr=target_fpr)
        print(f"🎯 Optimized threshold for {target_fpr:.1%} FPR: {optimal_threshold:.4f}")
    else:
        optimal_threshold = 0.5
        print("⚠️  No validation set provided, using default threshold 0.5")
    
    # Get test predictions
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            tabular = batch['tabular']
            cnn1d = batch['cnn1d']
            cnn2d = batch['cnn2d']
            labels = batch['label']
            
            outputs = model(tabular, cnn1d, cnn2d)
            
            test_preds.extend(outputs.cpu().numpy().flatten())
            test_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics with optimized threshold
    test_auc = roc_auc_score(test_labels, test_preds)
    test_preds_binary = (np.array(test_preds) > optimal_threshold).astype(int)
    test_accuracy = accuracy_score(test_labels, test_preds_binary)
    
    # Validate threshold performance
    if val_loader is not None:
        threshold_metrics = validate_threshold_performance(
            test_labels, test_preds, optimal_threshold, target_fpr=target_fpr
        )
    else:
        threshold_metrics = {}
    
    return {
        'auc': test_auc,
        'accuracy': test_accuracy,
        'optimal_threshold': optimal_threshold,
        'predictions': test_preds,
        'labels': test_labels,
        'predictions_binary': test_preds_binary,
        'threshold_metrics': threshold_metrics
    }

def analyze_cnn_coverage_impact(cnn_data: Dict, splits: Dict) -> Dict:
    """Analyze the impact of CNN data coverage on potential performance"""
    
    total_tabular = len(splits['train']['y']) + len(splits['val']['y']) + len(splits['test']['y'])
    total_cnn = len(cnn_data['kepids'])
    coverage_pct = (total_cnn / total_tabular) * 100
    
    # Analyze distribution by class
    confirmed_cnn = sum(1 for kepid in cnn_data['kepids'] if cnn_data['labels'][kepid] == 1)
    false_pos_cnn = total_cnn - confirmed_cnn
    
    analysis = {
        'total_tabular_samples': total_tabular,
        'total_cnn_samples': total_cnn,
        'coverage_percentage': coverage_pct,
        'confirmed_cnn_samples': confirmed_cnn,
        'false_positive_cnn_samples': false_pos_cnn,
        'residual_windows_count': len(cnn_data['residual_windows']),
        'pixel_diffs_count': len(cnn_data['pixel_diffs']),
    }
    
    print(f"\n📊 CNN COVERAGE ANALYSIS")
    print("=" * 30)
    print(f"📈 Total CNN coverage: {total_cnn}/{total_tabular} ({coverage_pct:.1f}%)")
    print(f"✅ Confirmed planets: {confirmed_cnn}")
    print(f"❌ False positives: {false_pos_cnn}")
    print(f"📡 Residual windows: {analysis['residual_windows_count']}")
    print(f"🖼️  Pixel differences: {analysis['pixel_diffs_count']}")
    
    # Recommendations
    if coverage_pct < 50:
        print(f"\n💡 RECOMMENDATION: Increase CNN coverage to >50% for better fusion performance")
        print(f"   Run: python scripts/generate_cnn_data_batch.py --max-targets 2000")
    elif coverage_pct < 80:
        print(f"\n💡 RECOMMENDATION: Good coverage! Consider expanding to >80% for optimal results")
    else:
        print(f"\n🎊 EXCELLENT: High CNN coverage should enable strong fusion performance!")
    
    return analysis

def main():
    """Enhanced main training pipeline with coverage analysis"""
    
    print("🌟 ENHANCED MULTI-MODAL EXOPLANET DETECTION PIPELINE")
    print("=" * 65)
    
    # Load tabular data
    print("\n📊 Loading tabular data...")
    data = load_and_prepare_data()
    splits = create_train_val_test_splits(data)
    feature_names = splits['feature_names']
    print(f"✅ Tabular features: {len(feature_names)}")
    
    # Load CNN data
    print("\n🧠 Loading CNN data...")
    cnn_data = load_cnn_data()
    
    # Analyze CNN coverage impact
    coverage_analysis = analyze_cnn_coverage_impact(cnn_data, splits)
    
    # Create aligned CNN datasets
    print("\n🔗 Creating aligned datasets...")
    cnn_datasets = create_cnn_datasets(cnn_data, splits, feature_names)
    
    # Create enhanced data loaders
    min_train_samples = min(len(splits['train']['y']), len(cnn_datasets['train']['y']))
    min_val_samples = min(len(splits['val']['y']), len(cnn_datasets['val']['y']))
    min_test_samples = min(len(splits['test']['y']), len(cnn_datasets['test']['y']))
    
    print(f"🎯 Multi-modal dataset sizes:")
    print(f"   Train: {min_train_samples} samples")
    print(f"   Val: {min_val_samples} samples")
    print(f"   Test: {min_test_samples} samples")
    
    # Create datasets with device placement
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = MultiModalDataset(
        splits['train']['X'][:min_train_samples],
        cnn_datasets['train']['cnn1d'][:min_train_samples],
        cnn_datasets['train']['cnn2d'][:min_train_samples],
        splits['train']['y'][:min_train_samples],
        device='cpu'  # Keep on CPU for DataLoader
    )
    
    val_dataset = MultiModalDataset(
        splits['val']['X'][:min_val_samples],
        cnn_datasets['val']['cnn1d'][:min_val_samples],
        cnn_datasets['val']['cnn2d'][:min_val_samples],
        splits['val']['y'][:min_val_samples],
        device='cpu'
    )
    
    test_dataset = MultiModalDataset(
        splits['test']['X'][:min_test_samples],
        cnn_datasets['test']['cnn1d'][:min_test_samples],
        cnn_datasets['test']['cnn2d'][:min_test_samples],
        splits['test']['y'][:min_test_samples],
        device='cpu'
    )
    
    # Adaptive batch size based on dataset size
    batch_size = min(32, max(8, min_train_samples // 10))
    print(f"📦 Using batch size: {batch_size}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create and train enhanced model
    print(f"\n🚀 Creating enhanced fusion model...")
    model = EnhancedMultiModalFusionModel(n_tabular_features=len(feature_names))
    
    training_results = train_enhanced_multimodal_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=150,
        learning_rate=1e-3,
        patience=20
    )
    
    # Evaluate with threshold optimization
    print(f"\n🎯 Evaluating with threshold optimization...")
    test_results = evaluate_with_threshold_optimization(
        model=training_results['model'],
        test_loader=test_loader,
        val_loader=val_loader,
        target_fpr=0.05
    )
    
    # Final results
    print("\n🏆 ENHANCED MULTI-MODAL RESULTS")
    print("=" * 45)
    print(f"🎯 Test AUC: {test_results['auc']:.4f}")
    print(f"🎯 Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"📈 Best Val AUC: {training_results['best_val_auc']:.4f}")
    print(f"🎚️  Optimal Threshold: {test_results['optimal_threshold']:.4f}")
    print(f"⏱️  Training Time: {training_results['training_time']:.1f}s")
    
    # CNN coverage impact
    print(f"\n📊 CNN Coverage Impact:")
    print(f"   Coverage: {coverage_analysis['coverage_percentage']:.1f}%")
    print(f"   CNN Samples: {coverage_analysis['total_cnn_samples']}")
    
    # Save enhanced model
    model_path = "models/enhanced_multimodal_fusion_model.pth"
    torch.save(training_results['model'].state_dict(), model_path)
    print(f"💾 Enhanced model saved to {model_path}")
    
    return training_results, test_results, coverage_analysis

if __name__ == "__main__":
    main()