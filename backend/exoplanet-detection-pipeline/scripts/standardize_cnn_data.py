"""
CNN data preprocessing to standardize shapes for training
"""
import numpy as np
import os
from scipy.ndimage import zoom
import glob

def standardize_residual_windows(input_dir="data/processed/residual_windows", 
                                output_dir="data/processed/residual_windows_std",
                                target_length=128):
    """Standardize residual windows to fixed length."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🔧 STANDARDIZING RESIDUAL WINDOWS TO LENGTH {target_length}")
    
    files = glob.glob(os.path.join(input_dir, "*.npy"))
    processed = 0
    
    for file_path in files:
        filename = os.path.basename(file_path)
        data = np.load(file_path)
        
        if len(data.shape) != 2:
            print(f"   ❌ Skip {filename}: unexpected shape {data.shape}")
            continue
            
        n_windows, current_length = data.shape
        
        if current_length == target_length:
            # Already correct length
            standardized = data
        elif current_length > target_length:
            # Downsample to target length
            indices = np.linspace(0, current_length-1, target_length, dtype=int)
            standardized = data[:, indices]
        else:
            # Upsample to target length (interpolate)
            zoom_factor = target_length / current_length
            standardized = zoom(data, (1, zoom_factor), order=1)
        
        # Ensure exactly target length
        if standardized.shape[1] != target_length:
            standardized = standardized[:, :target_length]
        
        # Save standardized data
        output_path = os.path.join(output_dir, filename)
        np.save(output_path, standardized.astype(np.float32))
        
        print(f"   ✅ {filename}: {data.shape} → {standardized.shape}")
        processed += 1
    
    print(f"📊 Processed {processed} residual window files")
    return processed

def standardize_pixel_diffs(input_dir="data/processed/pixel_diffs",
                           output_dir="data/processed/pixel_diffs_std", 
                           target_shape=(32, 24, 24)):
    """Standardize pixel differences to fixed shape."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🔧 STANDARDIZING PIXEL DIFFS TO SHAPE {target_shape}")
    
    files = glob.glob(os.path.join(input_dir, "*.npy"))
    processed = 0
    
    for file_path in files:
        filename = os.path.basename(file_path)
        data = np.load(file_path)
        
        if len(data.shape) != 3:
            print(f"   ❌ Skip {filename}: unexpected shape {data.shape}")
            continue
            
        current_shape = data.shape
        target_phases, target_h, target_w = target_shape
        
        # Handle phase dimension
        if current_shape[0] < target_phases:
            # Pad with zeros if not enough phases
            pad_phases = target_phases - current_shape[0]
            data = np.pad(data, ((0, pad_phases), (0, 0), (0, 0)), mode='constant')
        elif current_shape[0] > target_phases:
            # Take first target_phases if too many
            data = data[:target_phases]
        
        # Resize spatial dimensions
        if data.shape[1:] != (target_h, target_w):
            zoom_factors = (1, target_h / data.shape[1], target_w / data.shape[2])
            data = zoom(data, zoom_factors, order=1)
        
        # Ensure exact target shape
        standardized = data[:target_phases, :target_h, :target_w]
        
        # Save standardized data
        output_path = os.path.join(output_dir, filename)
        np.save(output_path, standardized.astype(np.float32))
        
        print(f"   ✅ {filename}: {current_shape} → {standardized.shape}")
        processed += 1
    
    print(f"📊 Processed {processed} pixel diff files")
    return processed

if __name__ == "__main__":
    print("🚀 CNN DATA STANDARDIZATION")
    print("="*35)
    
    # Standardize both data types
    residual_count = standardize_residual_windows()
    pixel_count = standardize_pixel_diffs()
    
    print(f"\n✅ STANDARDIZATION COMPLETE:")
    print(f"   📊 Residual windows: {residual_count} files → data/processed/residual_windows_std/")
    print(f"   📊 Pixel differences: {pixel_count} files → data/processed/pixel_diffs_std/")
    print(f"   🎯 Ready for multi-modal CNN training!")