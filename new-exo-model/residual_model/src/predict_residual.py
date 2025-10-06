import numpy as np
import torch
import sys
import os
from models_residual import ResidualCNN1D, LightweightResidualCNN

def load_model(weights_path="models/residual_cnn_best_fold0.pt", model_type="full"):
    """Load trained model"""
    if model_type == "full":
        model = ResidualCNN1D(in_features=128, base_channels=64, max_length=512)
    else:
        model = LightweightResidualCNN(in_features=128, base_channels=32, max_length=512)
    
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model

def preprocess_residual(arr, max_length=512):
    """Preprocess residual array to model input format"""
    seq_len, n_features = arr.shape
    
    if seq_len > max_length:
        # Take center portion for inference
        start_idx = (seq_len - max_length) // 2
        arr = arr[start_idx:start_idx + max_length]
    elif seq_len < max_length:
        # Pad or repeat as needed
        if seq_len < max_length // 4:  # Very short - repeat
            repeats = (max_length // seq_len) + 1
            arr = np.tile(arr, (repeats, 1))[:max_length]
        else:  # Pad with zeros
            padding = np.zeros((max_length - seq_len, n_features), dtype=arr.dtype)
            arr = np.concatenate([arr, padding], axis=0)
    
    # Convert to tensor: [seq_len, features] -> [1, features, seq_len]
    x = torch.tensor(arr.T[None, ...], dtype=torch.float32)
    return x

def predict_single(residual_path, weights_path="models/residual_cnn_best_fold0.pt", model_type="full"):
    """Predict probability for a single residual file"""
    
    # Load residual data
    try:
        arr = np.load(residual_path)
    except Exception as e:
        print(f"Error loading {residual_path}: {e}")
        return None
    
    if arr.shape[0] < 10:
        print(f"Warning: {residual_path} has very short sequence ({arr.shape[0]} samples)")
        return 0.0  # Default to false positive for very short sequences
    
    # Load model
    try:
        model = load_model(weights_path, model_type)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Preprocess and predict
    x = preprocess_residual(arr)
    
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).item()
    
    return prob

def predict_batch(residual_paths, weights_path="models/residual_cnn_best_fold0.pt", 
                  model_type="full", batch_size=32):
    """Predict probabilities for multiple residual files"""
    
    model = load_model(weights_path, model_type)
    results = []
    
    for i in range(0, len(residual_paths), batch_size):
        batch_paths = residual_paths[i:i+batch_size]
        batch_x = []
        valid_indices = []
        
        for j, path in enumerate(batch_paths):
            try:
                arr = np.load(path)
                if arr.shape[0] >= 10:  # Only process reasonable sequences
                    x = preprocess_residual(arr)
                    batch_x.append(x)
                    valid_indices.append(j)
                else:
                    results.append((path, 0.0))  # Default for short sequences
            except Exception as e:
                print(f"Error loading {path}: {e}")
                results.append((path, None))
        
        if batch_x:
            batch_tensor = torch.cat(batch_x, dim=0)
            
            with torch.no_grad():
                logits = model(batch_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()
            
            # Add results in correct order
            for idx, prob in zip(valid_indices, probs):
                results.append((batch_paths[idx], prob))
    
    return results

def analyze_residual_file(residual_path, weights_path="models/residual_cnn_best_fold0.pt"):
    """Detailed analysis of a single residual file"""
    
    print(f"Analyzing: {residual_path}")
    print("-" * 50)
    
    # Load and examine data
    try:
        arr = np.load(residual_path)
        print(f"Shape: {arr.shape}")
        print(f"Sequence length: {arr.shape[0]}")
        print(f"Features: {arr.shape[1]}")
        print(f"Data range: [{arr.min():.6f}, {arr.max():.6f}]")
        print(f"Mean: {arr.mean():.6f}, Std: {arr.std():.6f}")
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Make prediction
    prob = predict_single(residual_path, weights_path)
    
    if prob is not None:
        print(f"\\nPrediction:")
        print(f"  Probability of being CONFIRMED: {prob:.4f}")
        print(f"  Prediction: {'CONFIRMED' if prob > 0.5 else 'FALSE POSITIVE'}")
        print(f"  Confidence: {abs(prob - 0.5) + 0.5:.3f}")
    
    return prob

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python predict_residual.py <residual_file.npy> [model_weights.pt]")
        print("  python predict_residual.py --analyze <residual_file.npy> [model_weights.pt]")
        print("  python predict_residual.py --batch <file1.npy> <file2.npy> ... [model_weights.pt]")
        return
    
    weights = "models/residual_cnn_best_fold0.pt"
    if len(sys.argv) > 2 and sys.argv[-1].endswith('.pt'):
        weights = sys.argv[-1]
        files = sys.argv[1:-1]
    else:
        files = sys.argv[1:]
    
    if not os.path.exists(weights):
        print(f"Model weights not found: {weights}")
        print("Please train the model first with: python src/train_residual.py")
        return
    
    if files[0] == "--analyze":
        if len(files) < 2:
            print("Please provide a residual file to analyze")
            return
        analyze_residual_file(files[1], weights)
    
    elif files[0] == "--batch":
        if len(files) < 2:
            print("Please provide residual files for batch prediction")
            return
        results = predict_batch(files[1:], weights)
        print("Batch Prediction Results:")
        print("-" * 50)
        for path, prob in results:
            if prob is not None:
                pred = "CONFIRMED" if prob > 0.5 else "FALSE POSITIVE"
                print(f"{os.path.basename(path):30s} -> {prob:.4f} ({pred})")
            else:
                print(f"{os.path.basename(path):30s} -> ERROR")
    
    else:
        # Single file prediction
        residual_file = files[0]
        prob = predict_single(residual_file, weights)
        
        if prob is not None:
            pred = "CONFIRMED" if prob > 0.5 else "FALSE POSITIVE"
            print(f"{residual_file} -> p={prob:.4f} ({pred})")
        else:
            print(f"Error processing {residual_file}")

if __name__ == "__main__":
    main()