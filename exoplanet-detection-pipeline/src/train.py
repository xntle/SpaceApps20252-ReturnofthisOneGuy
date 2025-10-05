import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
import logging
from typing import Dict, List, Tuple, Optional
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    from .models import create_models
    from .data_loader import create_train_val_test_splits
except ImportError:
    # Handle when run as script or imported directly
    from models import create_models
    from data_loader import create_train_val_test_splits

logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """Save model checkpoint."""
        self.best_weights = model.state_dict().copy()

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def prepare_data_loaders(splits: Dict, 
                        batch_size: int = 32,
                        num_workers: int = 0) -> Dict[str, DataLoader]:
    """
    Prepare PyTorch data loaders from split data.
    
    Args:
        splits: Train/val/test splits from create_train_val_test_splits
        batch_size: Batch size for training
        num_workers: Number of worker processes
        
    Returns:
        Dictionary of data loaders
    """
    loaders = {}
    
    for split_name in ['train', 'val', 'test']:
        if split_name in splits:
            data = splits[split_name]
            
            # Create tensors
            X_tensor = torch.FloatTensor(data['X'])
            y_tensor = torch.FloatTensor(data['y']).unsqueeze(1)
            
            # Create dataset and loader
            dataset = TensorDataset(X_tensor, y_tensor)
            shuffle = (split_name == 'train')
            
            loaders[split_name] = DataLoader(
                dataset, 
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
    
    return loaders

def train_single_model(model: nn.Module,
                      train_loader: DataLoader,
                      val_loader: DataLoader,
                      config: Dict,
                      device: torch.device) -> Dict:
    """
    Train a single PyTorch model.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        device: Device to train on
        
    Returns:
        Training history and metrics
    """
    logger.info(f"Training {model.__class__.__name__}")
    
    model = model.to(device)
    
    # Loss function
    if config.get('use_focal_loss', False):
        criterion = FocalLoss(alpha=config.get('focal_alpha', 1), 
                             gamma=config.get('focal_gamma', 2))
    else:
        # Use class weights for imbalanced data
        pos_weight = torch.tensor([config.get('pos_weight', 1.0)]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer
    optimizer_name = config.get('optimizer', 'adam')
    lr = config.get('learning_rate', 0.001)
    weight_decay = config.get('weight_decay', 1e-4)
    
    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    if config.get('use_scheduler', True):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
    else:
        scheduler = None
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.get('patience', 10),
        min_delta=config.get('min_delta', 1e-6)
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_auc': [],
        'val_auc': [],
        'learning_rates': []
    }
    
    n_epochs = config.get('n_epochs', 100)
    
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_losses = []
        train_preds = []
        train_targets = []
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs} [Train]', leave=False)
        
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            if hasattr(model, 'forward') and 'tabular_data' in model.forward.__code__.co_varnames:
                # Hybrid model - need to split data appropriately
                # For simplicity, assume all data is tabular for individual model training
                output = model(data, torch.empty(0), torch.empty(0))[0]
            else:
                output = model(data)
            
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            if config.get('clip_grad_norm', 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'])
            
            optimizer.step()
            
            train_losses.append(loss.item())
            train_preds.extend(torch.sigmoid(output).detach().cpu().numpy())
            train_targets.extend(target.detach().cpu().numpy())
            
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation phase
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{n_epochs} [Val]', leave=False)
            
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)
                
                if hasattr(model, 'forward') and 'tabular_data' in model.forward.__code__.co_varnames:
                    output = model(data, torch.empty(0), torch.empty(0))[0]
                else:
                    output = model(data)
                
                loss = criterion(output, target)
                
                val_losses.append(loss.item())
                val_preds.extend(torch.sigmoid(output).cpu().numpy())
                val_targets.extend(target.cpu().numpy())
                
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate metrics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        try:
            train_auc = roc_auc_score(train_targets, train_preds)
            val_auc = roc_auc_score(val_targets, val_preds)
        except ValueError:
            # Handle case where only one class is present
            train_auc = 0.5
            val_auc = 0.5
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Logging
        logger.info(f"Epoch {epoch+1:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                   f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")
        
        # Learning rate scheduling
        if scheduler:
            scheduler.step(val_loss)
        
        # Early stopping
        if early_stopping(val_loss, model):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    return history

def train_fusion_stacker(individual_outputs: Dict[str, np.ndarray],
                        targets: np.ndarray,
                        config: Dict) -> xgb.XGBClassifier:
    """
    Train XGBoost fusion stacker on individual model outputs.
    
    Args:
        individual_outputs: Dictionary of model name -> predictions
        targets: Target labels
        config: Training configuration
        
    Returns:
        Trained XGBoost model
    """
    logger.info("Training XGBoost fusion stacker")
    
    # Combine individual outputs
    features = []
    feature_names = []
    
    for model_name, preds in individual_outputs.items():
        if preds.ndim == 1:
            features.append(preds.reshape(-1, 1))
            feature_names.append(f"{model_name}_pred")
        else:
            features.append(preds)
            for i in range(preds.shape[1]):
                feature_names.append(f"{model_name}_feature_{i}")
    
    X_stack = np.hstack(features)
    
    # Configure XGBoost
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'n_estimators': config.get('xgb_n_estimators', 100),
        'max_depth': config.get('xgb_max_depth', 6),
        'learning_rate': config.get('xgb_learning_rate', 0.1),
        'subsample': config.get('xgb_subsample', 0.8),
        'colsample_bytree': config.get('xgb_colsample_bytree', 0.8),
        'random_state': config.get('random_state', 42),
        'verbosity': 0
    }
    
    # Handle class imbalance
    scale_pos_weight = np.sum(targets == 0) / np.sum(targets == 1)
    xgb_params['scale_pos_weight'] = scale_pos_weight
    
    # Train model
    stacker = xgb.XGBClassifier(**xgb_params)
    stacker.fit(X_stack, targets)
    
    # Feature importance
    importance = stacker.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    logger.info("Feature importance in fusion stacker:")
    for _, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    return stacker

def main_training_pipeline(config: Dict) -> Dict:
    """
    Main training pipeline for the hybrid exoplanet detection system.
    
    Args:
        config: Configuration dictionary with all training parameters
        
    Returns:
        Dictionary with trained models and results
    """
    logger.info("Starting main training pipeline")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_gpu', True) else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.get('random_state', 42))
    np.random.seed(config.get('random_state', 42))
    
    # Load and prepare data
    try:
        from .data_loader import load_and_prepare_data
    except ImportError:
        from data_loader import load_and_prepare_data
    
    data = load_and_prepare_data(config.get('data_dir', 'data/raw/'))
    splits = create_train_val_test_splits(data, 
                                         test_size=config.get('test_size', 0.2),
                                         val_size=config.get('val_size', 0.2),
                                         random_state=config.get('random_state', 42))
    
    # Create data loaders
    loaders = prepare_data_loaders(splits, 
                                  batch_size=config.get('batch_size', 32))
    
    # Auto-set class weight from training split for better imbalance handling
    pos = splits['train']['y'].sum()
    neg = len(splits['train']['y']) - pos
    config['pos_weight'] = float(neg / max(pos, 1))
    logger.info(f"Auto-computed pos_weight: {config['pos_weight']:.3f} (neg={neg}, pos={pos})")
    
    # Create models (only the ones we want to train)
    model_config = {
        'tabular_input_size': splits['train']['X'].shape[1],
        'dropout_rate': config.get('dropout_rate', 0.3)
    }
    models_to_train = config.get('model_types', ['tabular'])
    models = create_models(model_config, models_to_train)
    
    # Training results
    results = {
        'models': {},
        'histories': {},
        'splits': splits
    }
    
    # Train individual models
    individual_outputs = {'train': {}, 'val': {}, 'test': {}}
    
    models_to_train = config.get('model_types', ['tabular', 'cnn1d', 'cnn2d'])
    
    for model_name in models_to_train:
        if model_name in models:
            logger.info(f"Training {model_name} model")
            
            model = models[model_name]
            history = train_single_model(
                model, loaders['train'], loaders['val'], config, device
            )
            
            results['models'][model_name] = model
            results['histories'][model_name] = history
            
            # Generate predictions for stacking
            model.eval()
            with torch.no_grad():
                for split in ['train', 'val', 'test']:
                    if split in loaders:
                        preds = []
                        for data, _ in loaders[split]:
                            data = data.to(device)
                            
                            if hasattr(model, 'forward') and 'tabular_data' in model.forward.__code__.co_varnames:
                                output = model(data, torch.empty(0), torch.empty(0))[0]
                            else:
                                output = model(data)
                            
                            preds.extend(torch.sigmoid(output).cpu().numpy())
                        
                        individual_outputs[split][model_name] = np.array(preds)
    
    # Train fusion stacker if multiple models were trained
    if len(individual_outputs['train']) > 1 and config.get('train_fusion', True):
        logger.info("Training fusion stacker")
        
        stacker = train_fusion_stacker(
            individual_outputs['train'],
            splits['train']['y'],
            config
        )
        
        results['stacker'] = stacker
        results['individual_outputs'] = individual_outputs
    
    # Save models if requested
    if config.get('save_models', True):
        save_dir = config.get('model_save_dir', 'models/')
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, model in results['models'].items():
            save_path = os.path.join(save_dir, f"{model_name}_model.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved {model_name} model to {save_path}")
        
        if 'stacker' in results:
            import joblib
            stacker_path = os.path.join(save_dir, "fusion_stacker.pkl")
            joblib.dump(results['stacker'], stacker_path)
            logger.info(f"Saved fusion stacker to {stacker_path}")
    
    logger.info("Training pipeline completed successfully")
    return results

def plot_training_history(histories: Dict[str, Dict], save_path: Optional[str] = None):
    """Plot training histories for all models."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for model_name, history in histories.items():
        # Loss curves
        axes[0, 0].plot(history['train_loss'], label=f'{model_name} train')
        axes[0, 1].plot(history['val_loss'], label=f'{model_name} val')
        
        # AUC curves
        axes[1, 0].plot(history['train_auc'], label=f'{model_name} train')
        axes[1, 1].plot(history['val_auc'], label=f'{model_name} val')
    
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].set_title('Training AUC')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].set_title('Validation AUC')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AUC')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training history plot to {save_path}")
    
    return fig

if __name__ == "__main__":
    # Test training pipeline with minimal configuration
    logging.basicConfig(level=logging.INFO)
    
    config = {
        'data_dir': 'data/raw/',
        'batch_size': 16,
        'n_epochs': 5,  # Short for testing
        'learning_rate': 0.001,
        'models_to_train': ['tabular'],  # Only train tabular model for testing
        'save_models': False,
        'train_fusion': False,
        'use_gpu': False,
        'random_state': 42
    }
    
    try:
        results = main_training_pipeline(config)
        print("Training pipeline test successful!")
        
        if results['histories']:
            plot_training_history(results['histories'])
            print("Training history plotted successfully!")
            
    except Exception as e:
        print(f"Training pipeline test failed: {e}")
        logger.error(f"Training pipeline test failed: {e}", exc_info=True)