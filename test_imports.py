#!/usr/bin/env python3
"""
Test Script
===========

Test that all modules can be imported correctly.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from models import TransformerTrajectoryPredictor, LSTMTrajectoryPredictor
        print("✅ Models imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import models: {e}")
        return False
    
    try:
        from data import TrajectoryDataset, load_and_prepare_data
        print("✅ Data modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import data modules: {e}")
        return False
    
    try:
        from training import ModelTrainer, ModelComparisonFramework
        print("✅ Training modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import training modules: {e}")
        return False
    
    try:
        from visualization import TrainingVisualizer, PredictionVisualizer
        print("✅ Visualization modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import visualization modules: {e}")
        return False
    
    return True

def test_model_creation():
    """Test that models can be created"""
    print("\nTesting model creation...")
    
    try:
        from models import TransformerTrajectoryPredictor, LSTMTrajectoryPredictor
        
        # Create models
        transformer = TransformerTrajectoryPredictor()
        lstm = LSTMTrajectoryPredictor()
        
        print(f"✅ Transformer created with {sum(p.numel() for p in transformer.parameters()):,} parameters")
        print(f"✅ LSTM created with {sum(p.numel() for p in lstm.parameters()):,} parameters")
        
        return True
    except Exception as e:
        print(f"❌ Failed to create models: {e}")
        return False

def test_data_loading():
    """Test data loading"""
    print("\nTesting data loading...")
    
    try:
        from data import load_and_prepare_data
        
        # Try to load data (will generate sample data if file not found)
        data, norm_stats = load_and_prepare_data("nonexistent_file.npz")
        
        print(f"✅ Data loaded: {data.shape}")
        print(f"✅ Normalization stats: {list(norm_stats.keys())}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Vessel Trajectory Prediction Framework")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Model Creation", test_model_creation),
        ("Data Loading", test_data_loading)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed!")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The framework is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)