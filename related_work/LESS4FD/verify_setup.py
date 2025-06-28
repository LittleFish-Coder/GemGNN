#!/usr/bin/env python3
"""
Quick verification script for LESS4FD setup.
Run this to ensure everything is properly configured for few-shot training.
"""

def verify_setup():
    """Verify LESS4FD setup for few-shot training."""
    print("🔍 Verifying LESS4FD setup for few-shot training...")
    
    # Test imports
    try:
        from train_less4fd import LESS4FDTrainer
        from build_less4fd_graph import LESS4FDGraphBuilder
        from models.less4fd_model import LESS4FDModel
        from config.less4fd_config import LESS4FD_CONFIG, FEWSHOT_CONFIG
        print("✅ All core modules imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Check configuration
    try:
        k_shots = FEWSHOT_CONFIG["k_shot_range"]
        print(f"✅ Few-shot configuration loaded: k-shots {k_shots}")
    except KeyError as e:
        print(f"❌ Configuration error: {e}")
        return False
    
    print("✅ LESS4FD is ready for few-shot training!")
    print("\n📋 Quick usage:")
    print("python train_less4fd.py --dataset politifact --k_shot 5")
    print("python run_less4fd_experiments.py --single")
    print("python test_less4fd.py")
    
    return True

if __name__ == "__main__":
    import sys
    success = verify_setup()
    sys.exit(0 if success else 1)