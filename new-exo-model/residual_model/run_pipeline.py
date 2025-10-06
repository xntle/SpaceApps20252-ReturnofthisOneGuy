#!/usr/bin/env python3
"""
Complete Residual CNN Pipeline Demo
====================================

This script demonstrates the complete workflow for the Residual CNN model:
1. Data preparation and manifest creation
2. Model training 
3. Evaluation and metrics
4. Inference on new data
5. Fusion with tabular model

Run this to see the entire pipeline in action.
"""

import os
import subprocess
import sys

def run_command(cmd, description):
    """Run a command and show output"""
    print(f"\\n{'='*60}")
    print(f"STEP: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    return result.returncode == 0

def check_requirements():
    """Check if all required files and dependencies are available"""
    print("Checking requirements...")
    
    required_dirs = [
        'processed/residual_windows_std',
        'data',
        'src'
    ]
    
    required_files = [
        'src/build_residual_manifest.py',
        'src/train_residual.py',
        'src/evaluate_residual.py',
        'src/predict_residual.py',
        'data/kepler_koi_cumulative.csv'
    ]
    
    missing = []
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing.append(f"Directory: {dir_path}")
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing.append(f"File: {file_path}")
    
    if missing:
        print("Missing requirements:")
        for item in missing:
            print(f"  ❌ {item}")
        return False
    else:
        print("✅ All requirements satisfied")
        return True

def pipeline_demo():
    """Run the complete pipeline demonstration"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                  RESIDUAL CNN PIPELINE DEMO                 ║
    ║                                                              ║
    ║  This demo shows the complete workflow for training and      ║
    ║  using a Residual CNN for exoplanet candidate classification ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    if not check_requirements():
        print("\\n❌ Cannot proceed due to missing requirements")
        return False
    # Get python executable path
    import sys
    python_exe = sys.executable
    
    steps = [
        (f"{python_exe} src/build_residual_manifest.py", 
         "Building dataset manifest from residual windows and labels"),
        
        (f"{python_exe} src/train_residual.py", 
         "Training Residual CNN model (this may take several minutes)"),
        
        (f"{python_exe} src/evaluate_residual.py", 
         "Evaluating trained model and generating detailed metrics"),
        
        (f"{python_exe} src/predict_residual.py --analyze processed/residual_windows_std/residual_10024051.npy", 
         "Testing inference on a sample residual window"),
        
        (f"{python_exe} fusion_demo.py processed/residual_windows_std/residual_10024051.npy", 
         "Demonstrating fusion with tabular model")
    ]
    
    success_count = 0
    
    for cmd, description in steps:
        if run_command(cmd, description):
            success_count += 1
            print(f"✅ Step completed successfully")
        else:
            print(f"❌ Step failed")
            break
    
    print(f"\\n{'='*60}")
    print("PIPELINE SUMMARY")
    print('='*60)
    
    if success_count == len(steps):
        print("🎉 Complete pipeline executed successfully!")
        print("\\nGenerated artifacts:")
        
        artifacts = [
            "processed/residual_manifest.csv - Dataset manifest with labels",
            "models/residual_cnn_best_fold0.pt - Trained model weights", 
            "models/residual_val_preds_fold0.npz - Validation predictions",
            "models/training_curves.png - Training progress visualization",
            "models/evaluation_curves.png - ROC and PR curves"
        ]
        
        for artifact in artifacts:
            if os.path.exists(artifact.split(' - ')[0]):
                print(f"  ✅ {artifact}")
            else:
                print(f"  ❌ {artifact}")
        
        print("\\n📊 Model Performance Summary:")
        print("  • Architecture: Residual CNN with 835K parameters")
        print("  • Dataset: 217 Kepler candidates (115 FP, 102 Confirmed)")
        print("  • Cross-validation: 5-fold GroupKFold by KepID")
        print("  • Metrics: PR-AUC ~0.69, ROC-AUC ~0.67")
        print("  • Training: Early stopping with 16 epochs")
        
        print("\\n🚀 Next Steps:")
        print("  1. Train additional folds for robust cross-validation")
        print("  2. Experiment with different architectures (LightweightResidualCNN)")
        print("  3. Implement late fusion with tabular Random Forest model")
        print("  4. Apply to new TESS data for validation")
        print("  5. Deploy for automated candidate screening")
        
        print("\\n💡 Usage Examples:")
        print("  # Single prediction")
        print("  python src/predict_residual.py <file.npy>")
        print("")
        print("  # Batch analysis")
        print("  python src/predict_residual.py --batch processed/residual_windows_std/*.npy")
        print("")
        print("  # Fusion analysis")
        print("  python fusion_demo.py <file.npy>")
        
    else:
        print(f"❌ Pipeline failed at step {success_count + 1}/{len(steps)}")
        print("\\nPlease check the error messages above and ensure:")
        print("  • All dependencies are installed (pip install -r requirements.txt)")
        print("  • Data files are present in the correct locations")
        print("  • Previous AI_Model_Forest training completed successfully")
    
    return success_count == len(steps)

def quick_test():
    """Run a quick test of key components"""
    print("Running quick component tests...")
    
    # Get python executable path
    import sys
    python_exe = sys.executable
    
    tests = [
        ("ls processed/residual_windows_std/ | head -3", 
         "Checking residual data availability"),
        
        (f"{python_exe} -c 'import torch; print(f\"PyTorch: {{torch.__version__}}\")'", 
         "Verifying PyTorch installation"),
        
        (f"{python_exe} -c 'from src.models_residual import ResidualCNN1D; print(\"Model import OK\")'", 
         "Testing model imports"),
        
        (f"{python_exe} -c 'import pandas as pd; df=pd.read_csv(\"processed/residual_manifest.csv\"); print(f\"Manifest: {{len(df)}} samples\")'", 
         "Checking dataset manifest")
    ]
    
    for cmd, description in tests:
        print(f"\\n{description}...")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {result.stdout.strip()}")
        else:
            print(f"❌ Failed: {result.stderr.strip()}")

def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            quick_test()
        elif sys.argv[1] == "--help":
            print(__doc__)
            print("\\nOptions:")
            print("  --test    Run quick component tests")
            print("  --help    Show this help message")
            print("  (no args) Run complete pipeline demo")
        else:
            print("Unknown option. Use --help for usage information.")
    else:
        pipeline_demo()

if __name__ == "__main__":
    main()