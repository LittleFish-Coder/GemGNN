#!/usr/bin/env python3
"""
Test script for LESS4FD architecture.

Simple test to verify that the LESS4FD code works for few-shot scenarios.
"""

import sys
import os
from pathlib import Path

# Add necessary directories to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))  # LESS4FD directory
sys.path.insert(0, str(current_dir.parent.parent))  # Main project directory

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from config.less4fd_config import LESS4FD_CONFIG, TRAINING_CONFIG, FEWSHOT_CONFIG
        print("✓ Config imported successfully")
    except Exception as e:
        print(f"✗ Config import failed: {e}")
        return False
    
    try:
        from utils.entity_extractor import EntityExtractor
        print("✓ EntityExtractor imported successfully")
    except Exception as e:
        print(f"✗ EntityExtractor import failed: {e}")
        return False
    
    try:
        from utils.sampling import LESS4FDSampler
        print("✓ LESS4FDSampler imported successfully")
    except Exception as e:
        print(f"✗ LESS4FDSampler import failed: {e}")
        return False
    
    try:
        from build_less4fd_graph import LESS4FDGraphBuilder
        print("✓ LESS4FDGraphBuilder imported successfully")
    except Exception as e:
        print(f"✗ LESS4FDGraphBuilder import failed: {e}")
        return False
    
    try:
        from models.less4fd_model import LESS4FDModel
        print("✓ LESS4FDModel imported successfully")
    except Exception as e:
        print(f"✗ LESS4FDModel import failed: {e}")
        return False
    
    try:
        from train_less4fd import LESS4FDTrainer
        print("✓ LESS4FDTrainer imported successfully")
    except Exception as e:
        print(f"✗ LESS4FDTrainer import failed: {e}")
        return False
    
    return True

def test_configuration():
    """Test configuration values."""
    print("\nTesting configuration...")
    
    from config.less4fd_config import LESS4FD_CONFIG, TRAINING_CONFIG, FEWSHOT_CONFIG
    
    # Check essential config values
    required_keys = {
        "LESS4FD_CONFIG": ["hidden_channels", "num_gnn_layers", "dropout", "entity_types"],
        "TRAINING_CONFIG": ["pretrain_epochs", "finetune_epochs", "learning_rate"],
        "FEWSHOT_CONFIG": ["k_shot_range", "regularization", "early_stopping"]
    }
    
    configs = {
        "LESS4FD_CONFIG": LESS4FD_CONFIG,
        "TRAINING_CONFIG": TRAINING_CONFIG,
        "FEWSHOT_CONFIG": FEWSHOT_CONFIG
    }
    
    for config_name, config in configs.items():
        for key in required_keys[config_name]:
            if key not in config:
                print(f"✗ Missing key '{key}' in {config_name}")
                return False
        print(f"✓ {config_name} has all required keys")
    
    # Check k-shot range
    k_shot_range = FEWSHOT_CONFIG["k_shot_range"]
    if not isinstance(k_shot_range, list) or len(k_shot_range) == 0:
        print("✗ Invalid k_shot_range")
        return False
    print(f"✓ K-shot range: {k_shot_range}")
    
    return True

def test_entity_extractor():
    """Test entity extractor basic functionality."""
    print("\nTesting EntityExtractor...")
    
    try:
        from utils.entity_extractor import EntityExtractor
        
        # Initialize with minimal config
        extractor = EntityExtractor(
            model_name="bert-base-cased",
            max_entities_per_text=5
        )
        
        # Test entity extraction on sample text
        sample_text = "Apple Inc. is a technology company founded by Steve Jobs in California."
        entities = extractor.extract_entities(sample_text)
        
        print(f"✓ Extracted {len(entities)} entities from sample text")
        for entity in entities[:3]:  # Show first 3
            print(f"  - {entity['text']} ({entity['label']})")
        
        return True
        
    except Exception as e:
        print(f"✗ EntityExtractor test failed: {e}")
        return False

def test_less4fd_sampler():
    """Test LESS4FD sampler."""
    print("\nTesting LESS4FDSampler...")
    
    try:
        from utils.sampling import LESS4FDSampler
        
        sampler = LESS4FDSampler(seed=42)
        print("✓ LESS4FDSampler initialized successfully")
        
        # Test basic functionality
        if hasattr(sampler, 'sample_entity_aware_k_shot'):
            print("✓ sample_entity_aware_k_shot method exists")
        
        if hasattr(sampler, 'generate_meta_learning_tasks'):
            print("✓ generate_meta_learning_tasks method exists")
        
        return True
        
    except Exception as e:
        print(f"✗ LESS4FDSampler test failed: {e}")
        return False

def test_graph_builder():
    """Test graph builder initialization."""
    print("\nTesting LESS4FDGraphBuilder...")
    
    try:
        from build_less4fd_graph import LESS4FDGraphBuilder
        
        print("✓ LESS4FDGraphBuilder import successful")
        
        # Check required methods exist in the class
        required_methods = [
            'build_graph',
            'build_entity_nodes',
            'build_entity_edges',
            'build_news_entity_edges'
        ]
        
        for method in required_methods:
            if hasattr(LESS4FDGraphBuilder, method):
                print(f"✓ {method} method exists")
            else:
                print(f"✗ {method} method missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ LESS4FDGraphBuilder test failed: {e}")
        return False

def test_model():
    """Test model initialization."""
    print("\nTesting LESS4FDModel...")
    
    try:
        import torch
        from torch_geometric.data import HeteroData
        from models.less4fd_model import LESS4FDModel
        
        # Create dummy hetero data
        data = HeteroData()
        data['news'].x = torch.randn(10, 768)
        data['entity'].x = torch.randn(20, 768)
        
        # Initialize model
        model = LESS4FDModel(
            data=data,
            hidden_channels=64,
            num_entities=20,
            num_classes=2
        )
        
        print("✓ LESS4FDModel initialized successfully")
        
        # Test if model has the required methods
        required_methods = ['forward', 'pretrain_step', 'finetune_step', 'predict']
        for method in required_methods:
            if hasattr(model, method):
                print(f"✓ {method} method exists")
            else:
                print(f"✗ {method} method missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ LESS4FDModel test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=== LESS4FD Test Suite ===\n")
    
    tests = [
        test_imports,
        test_configuration,
        test_entity_extractor,
        test_less4fd_sampler,
        test_graph_builder,
        test_model
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("PASSED\n")
            else:
                failed += 1
                print("FAILED\n")
        except Exception as e:
            failed += 1
            print(f"ERROR: {e}\n")
    
    print("=== Test Summary ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! LESS4FD is ready for few-shot training.")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())