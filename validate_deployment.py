#!/usr/bin/env python3
"""
Comprehensive validation script for Wan2.2 RunPod serverless deployment.
This script tests all imports, configurations, and basic functionality.
"""

import sys
import os
import traceback
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "builder"))

def test_imports():
    """Test all required imports."""
    print("🧪 Testing imports...")
    
    try:
        # Test core PyTorch imports
        import torch
        print(f"✅ PyTorch {torch.__version__} imported successfully")
        
        # Test diffusers imports
        from diffusers import WanPipeline
        from diffusers.utils import export_to_video
        print("✅ Diffusers imports successful")
        
        # Test RunPod import
        import runpod
        print("✅ RunPod SDK imported successfully")
        
        # Test our custom modules
        from src.utils import validate_input, save_video, get_memory_usage
        print("✅ Utils module imported successfully")
        
        from src.predict import Wan22VideoGenerator, generate_video
        print("✅ Predict module imported successfully")
        
        from src.handler import handler, health_check
        print("✅ Handler module imported successfully")
        
        from builder.download_model import download_model, verify_model
        print("✅ Download model module imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_parameter_validation():
    """Test parameter validation."""
    print("\n🧪 Testing parameter validation...")
    
    try:
        from src.utils import validate_input
        
        # Test valid input
        valid_input = {
            "prompt": "A cat walking in a park",
            "negative_prompt": "blurry",
            "num_frames": 81,
            "guidance_scale": 6.0,
            "num_inference_steps": 40,
            "width": 720,
            "height": 480,
            "fps": 16,
            "seed": 42
        }
        
        params = validate_input(valid_input)
        print("✅ Valid input validation passed")
        
        # Test missing prompt
        try:
            invalid_input = {"num_frames": 81}
            validate_input(invalid_input)
            print("❌ Should have failed on missing prompt")
            return False
        except ValueError:
            print("✅ Missing prompt validation works")
        
        # Test parameter ranges
        boundary_input = {
            "prompt": "test",
            "num_frames": 300,  # Out of range
            "guidance_scale": 25.0,  # Out of range
        }
        
        params = validate_input(boundary_input)
        assert params["num_frames"] == 81  # Should be corrected to default
        assert params["guidance_scale"] == 5.0  # Should be corrected to default
        print("✅ Parameter range validation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Parameter validation failed: {e}")
        traceback.print_exc()
        return False

def test_model_configuration():
    """Test model configuration and paths."""
    print("\n🧪 Testing model configuration...")
    
    try:
        from src.handler import MODEL_ID, MODEL_DIR, OUTPUT_DIR
        
        # Check model ID
        expected_model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        if MODEL_ID == expected_model_id:
            print(f"✅ Model ID correct: {MODEL_ID}")
        else:
            print(f"❌ Model ID incorrect: {MODEL_ID} != {expected_model_id}")
            return False
        
        # Check paths
        if MODEL_DIR == "/runpod-volume/model":
            print(f"✅ Model directory correct: {MODEL_DIR}")
        else:
            print(f"❌ Model directory incorrect: {MODEL_DIR}")
            return False
            
        if OUTPUT_DIR == "/runpod-volume/outputs":
            print(f"✅ Output directory correct: {OUTPUT_DIR}")
        else:
            print(f"❌ Output directory incorrect: {OUTPUT_DIR}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Model configuration test failed: {e}")
        traceback.print_exc()
        return False

def test_json_inputs():
    """Test JSON input parsing."""
    print("\n🧪 Testing JSON input files...")
    
    try:
        import json
        
        # Test simple input file
        simple_input_path = Path(__file__).parent / "test_input.json"
        if simple_input_path.exists():
            with open(simple_input_path, 'r') as f:
                data = json.load(f)
            
            if "input" in data and "prompt" in data["input"]:
                print("✅ test_input.json is valid")
            else:
                print("❌ test_input.json missing required fields")
                return False
        else:
            print("⚠️ test_input.json not found")
        
        # Test complex input file
        complex_input_path = Path(__file__).parent / "test_input_complex.json"
        if complex_input_path.exists():
            with open(complex_input_path, 'r') as f:
                data = json.load(f)
            
            if "input" in data and "prompt" in data["input"]:
                print("✅ test_input_complex.json is valid")
            else:
                print("❌ test_input_complex.json missing required fields")
                return False
        else:
            print("⚠️ test_input_complex.json not found")
        
        return True
        
    except Exception as e:
        print(f"❌ JSON input test failed: {e}")
        traceback.print_exc()
        return False

def test_handler_structure():
    """Test handler function structure."""
    print("\n🧪 Testing handler structure...")
    
    try:
        from src.handler import handler, health_check
        
        # Test handler signature
        test_event = {
            "input": {
                "prompt": "test prompt",
                "num_frames": 81
            }
        }
        
        # This should not actually run inference, just test structure
        print("✅ Handler function accessible")
        
        # Test health check
        health_result = health_check()
        if isinstance(health_result, dict) and "status" in health_result:
            print("✅ Health check function works")
        else:
            print("❌ Health check function invalid")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Handler structure test failed: {e}")
        traceback.print_exc()
        return False

def test_docker_requirements():
    """Test Docker-related requirements."""
    print("\n🧪 Testing Docker requirements...")
    
    try:
        # Check requirements.txt exists
        req_path = Path(__file__).parent / "builder" / "requirements.txt"
        if req_path.exists():
            print("✅ requirements.txt exists")
            
            # Check key dependencies
            with open(req_path, 'r') as f:
                content = f.read()
            
            required_packages = [
                "torch>=2.4.0",
                "diffusers",
                "runpod",
                "transformers",
                "huggingface_hub"
            ]
            
            missing_packages = []
            for package in required_packages:
                if package.split(">=")[0].split("==")[0] not in content:
                    missing_packages.append(package)
            
            if missing_packages:
                print(f"❌ Missing packages in requirements.txt: {missing_packages}")
                return False
            else:
                print("✅ All required packages in requirements.txt")
        else:
            print("❌ requirements.txt not found")
            return False
        
        # Check Dockerfile exists
        dockerfile_path = Path(__file__).parent / "Dockerfile"
        if dockerfile_path.exists():
            print("✅ Dockerfile exists")
        else:
            print("❌ Dockerfile not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Docker requirements test failed: {e}")
        traceback.print_exc()
        return False

def test_memory_functions():
    """Test memory monitoring functions."""
    print("\n🧪 Testing memory functions...")
    
    try:
        from src.utils import get_memory_usage, format_duration
        
        # Test memory usage (should work even without GPU)
        memory_info = get_memory_usage()
        if isinstance(memory_info, dict):
            print("✅ Memory usage function works")
        else:
            print("❌ Memory usage function failed")
            return False
        
        # Test duration formatting
        duration_str = format_duration(125.5)
        if isinstance(duration_str, str) and "m" in duration_str:
            print("✅ Duration formatting works")
        else:
            print("❌ Duration formatting failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Memory functions test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all validation tests."""
    print("🚀 Starting Wan2.2 RunPod Deployment Validation\n")
    
    tests = [
        ("Imports", test_imports),
        ("Parameter Validation", test_parameter_validation),
        ("Model Configuration", test_model_configuration),
        ("JSON Inputs", test_json_inputs),
        ("Handler Structure", test_handler_structure),
        ("Docker Requirements", test_docker_requirements),
        ("Memory Functions", test_memory_functions),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Testing: {test_name}")
        print(f"{'='*50}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("VALIDATION SUMMARY")
    print(f"{'='*50}")
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {passed + failed} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Deployment package is ready.")
        return True
    else:
        print(f"\n⚠️ {failed} tests failed. Please fix issues before deployment.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 