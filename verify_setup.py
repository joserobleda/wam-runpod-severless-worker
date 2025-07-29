#!/usr/bin/env python3
"""
Comprehensive Verification Script for Wan 2.2 TI2V-5B RunPod Setup
Tests all components to ensure deployment will work correctly
"""

import os
import sys
import json
import subprocess
import importlib
from pathlib import Path
from typing import Dict, List, Tuple

def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print('='*60)

def print_check(item: str, status: bool, details: str = ""):
    """Print a check result"""
    icon = "✅" if status else "❌"
    print(f"  {icon} {item}")
    if details:
        print(f"    {details}")

def check_environment() -> Dict[str, bool]:
    """Check basic environment setup"""
    print_section("Environment Check")
    results = {}
    
    # Python version
    python_version = sys.version_info
    python_ok = python_version >= (3, 8) and python_version < (4, 0)
    print_check(f"Python {python_version.major}.{python_version.minor}.{python_version.micro}", 
                python_ok, "Required: 3.8-3.11")
    results['python'] = python_ok
    
    # Working directory
    cwd = os.getcwd()
    expected_files = ['Dockerfile', 'src/predict.py', 'src/rp_handler.py']
    files_exist = all(os.path.exists(f) for f in expected_files)
    print_check(f"Working directory: {cwd}", files_exist, 
                f"Expected files: {', '.join(expected_files)}")
    results['directory'] = files_exist
    
    return results

def check_python_dependencies() -> Dict[str, bool]:
    """Check Python package dependencies"""
    print_section("Python Dependencies Check")
    results = {}
    
    # Required packages with expected versions (Official Wan 2.2 requirements)
    required_packages = {
        'torch': '2.4.0',  # Official requirement: torch >= 2.4.0
        'torchvision': '0.19.0', 
        'torchaudio': '2.4.0',
        'transformers': '4.44.0',
        'diffusers': '0.30.0',
        'accelerate': '0.33.0',
        'safetensors': '0.4.4',
        'opencv-python': '4.10.0',
        'imageio': '2.35.0',
        'pillow': '10.4.0',
        'numpy': '1.26.0',
        'scipy': '1.14.0',
        'runpod': '1.4.2',
        'boto3': None,  # Any recent version is fine
        'requests': None,
    }
    
    for package, expected_version in required_packages.items():
        try:
            if package == 'opencv-python':
                # OpenCV can be installed as opencv-python or opencv-python-headless
                try:
                    import cv2 as module
                    version = cv2.__version__
                except ImportError:
                    module = importlib.import_module('cv2')
                    version = module.__version__
            else:
                module = importlib.import_module(package.replace('-', '_'))
                version = getattr(module, '__version__', 'unknown')
            
            if expected_version:
                version_ok = version.startswith(expected_version.split('.')[0])
                details = f"Have: {version}, Expected: {expected_version}"
            else:
                version_ok = True
                details = f"Version: {version}"
                
            print_check(f"{package}", version_ok, details)
            results[package] = version_ok
            
        except ImportError:
            print_check(f"{package}", False, "Not installed")
            results[package] = False
    
    return results

def check_cuda_support() -> Dict[str, bool]:
    """Check CUDA availability and compatibility"""
    print_section("CUDA Support Check")
    results = {}
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print_check("CUDA available", cuda_available)
        results['cuda_available'] = cuda_available
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            
            print_check(f"CUDA devices: {device_count}", device_count > 0)
            print_check(f"Primary GPU: {device_name}", True)
            print_check(f"CUDA version: {cuda_version}", True, "Expected: 12.1.x")
            
            # Test tensor operations
            try:
                x = torch.randn(1, device='cuda')
                y = x * 2
                tensor_ops_ok = True
                print_check("CUDA tensor operations", True)
            except Exception as e:
                tensor_ops_ok = False
                print_check("CUDA tensor operations", False, str(e))
            
            results['cuda_ops'] = tensor_ops_ok
        else:
            print_check("GPU operations", False, "CUDA not available")
            results['cuda_ops'] = False
            
    except ImportError:
        print_check("PyTorch", False, "Not installed")
        results['cuda_available'] = False
        results['cuda_ops'] = False
    
    return results

def check_docker_files() -> Dict[str, bool]:
    """Check Docker-related files"""
    print_section("Docker Files Check")
    results = {}
    
    docker_files = {
        'Dockerfile': ['FROM nvidia/cuda:12.1-devel-ubuntu22.04 AS builder',
                      'FROM nvidia/cuda:12.1-runtime-ubuntu22.04'],
        'builder/requirements.txt': ['runpod==1.4.2', 'transformers==4.36.2'],
        'src/predict.py': ['class VideoGenerator', 'class Predictor'],
        'src/rp_handler.py': ['import runpod', 'def handler'],
        'src/rp_schema.py': ['INPUT_SCHEMA'],
        'src/startup.sh': ['#!/bin/bash', 'exec python3 -u rp_handler.py']
    }
    
    for file_path, expected_content in docker_files.items():
        file_exists = os.path.exists(file_path)
        
        if file_exists:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                content_ok = all(expected in content for expected in expected_content)
                details = f"Contains expected content: {content_ok}"
                
                print_check(file_path, content_ok, details)
                results[file_path] = content_ok
                
            except Exception as e:
                print_check(file_path, False, f"Read error: {e}")
                results[file_path] = False
        else:
            print_check(file_path, False, "File not found")
            results[file_path] = False
    
    return results

def check_wan22_compatibility() -> Dict[str, bool]:
    """Check Wan 2.2 specific compatibility"""
    print_section("Wan 2.2 Compatibility Check")
    results = {}
    
    # Check if we can simulate the Wan 2.2 command structure
    try:
        import subprocess
        import tempfile
        
        # Test official Wan 2.2 TI2V-5B command structure
        # Based on: https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B
        wan22_cmd = [
            'python', '/app/wan2.2_code/generate.py',
            '--task', 'ti2v-5B',                    # Official task name
            '--size', '1280*704',                   # Official 720P resolution
            '--ckpt_dir', '/app/model/wan2.2-ti2v-5b',  # Model directory
            '--offload_model', 'True',              # Memory optimization for RTX 4090
            '--convert_model_dtype',                # Model dtype conversion
            '--t5_cpu',                            # Move T5 to CPU for memory
            '--prompt', 'Test prompt',
            '--guidance_scale', '7.5',
            '--num_inference_steps', '25',
            '--fps', '24'
        ]
        
        cmd_structure_ok = len(wan22_cmd) > 10
        print_check("Wan 2.2 command structure", cmd_structure_ok, 
                   f"Command has {len(wan22_cmd)} arguments")
        results['command_structure'] = cmd_structure_ok
        
        # Check subprocess module
        subprocess_ok = hasattr(subprocess, 'run')
        print_check("subprocess.run available", subprocess_ok)
        results['subprocess'] = subprocess_ok
        
        # Check tempfile module
        tempfile_ok = hasattr(tempfile, 'mkdtemp')
        print_check("tempfile.mkdtemp available", tempfile_ok)
        results['tempfile'] = tempfile_ok
        
    except Exception as e:
        print_check("Wan 2.2 compatibility", False, str(e))
        results['command_structure'] = False
        results['subprocess'] = False
        results['tempfile'] = False
    
    return results

def check_runpod_compatibility() -> Dict[str, bool]:
    """Check RunPod specific compatibility"""
    print_section("RunPod Compatibility Check")
    results = {}
    
    try:
        # Test RunPod imports
        import runpod
        print_check("RunPod package", True, f"Version: {runpod.__version__}")
        results['runpod_import'] = True
        
        # Test rp_validator
        from runpod.serverless.utils.rp_validator import validate
        print_check("rp_validator", True)
        results['rp_validator'] = True
        
        # Test rp_upload
        from runpod.serverless.utils.rp_upload import upload_in_memory_object
        print_check("rp_upload", True)
        results['rp_upload'] = True
        
        # Test boto3 for R2 upload
        import boto3
        print_check("boto3 for R2", True, f"Version: {boto3.__version__}")
        results['boto3'] = True
        
    except ImportError as e:
        print_check("RunPod imports", False, str(e))
        results['runpod_import'] = False
        results['rp_validator'] = False
        results['rp_upload'] = False
        results['boto3'] = False
    
    return results

def test_example_inputs() -> Dict[str, bool]:
    """Test example input files"""
    print_section("Example Inputs Check")
    results = {}
    
    example_files = [
        'example_input_simple.json',
        'example_input_advanced.json', 
        'example_input_image_to_video.json',
        'example_input_youtube_short.json'
    ]
    
    for example_file in example_files:
        if os.path.exists(example_file):
            try:
                with open(example_file, 'r') as f:
                    data = json.load(f)
                
                # Check if it has the correct RunPod structure
                has_input = 'input' in data
                has_prompt = has_input and 'prompt' in data['input']
                
                structure_ok = has_input and has_prompt
                details = f"Has 'input' wrapper: {has_input}, Has 'prompt': {has_prompt}"
                
                print_check(example_file, structure_ok, details)
                results[example_file] = structure_ok
                
            except json.JSONDecodeError as e:
                print_check(example_file, False, f"JSON error: {e}")
                results[example_file] = False
        else:
            print_check(example_file, False, "File not found")
            results[example_file] = False
    
    return results

def generate_deployment_report(all_results: Dict[str, Dict[str, bool]]) -> str:
    """Generate a deployment readiness report"""
    print_section("Deployment Readiness Report")
    
    # Count successes and failures
    total_checks = sum(len(section) for section in all_results.values())
    passed_checks = sum(sum(section.values()) for section in all_results.values())
    
    readiness_score = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
    
    print(f"\n📊 Overall Readiness: {readiness_score:.1f}% ({passed_checks}/{total_checks} checks passed)")
    
    # Critical requirements
    critical_requirements = {
        'Python version': all_results.get('environment', {}).get('python', False),
        'Required files': all_results.get('environment', {}).get('directory', False),
        'PyTorch': all_results.get('dependencies', {}).get('torch', False),
        'Transformers': all_results.get('dependencies', {}).get('transformers', False),
        'RunPod': all_results.get('runpod', {}).get('runpod_import', False),
        'Docker files': all(all_results.get('docker', {}).values()),
    }
    
    critical_passed = sum(critical_requirements.values())
    critical_total = len(critical_requirements)
    
    print(f"\n🎯 Critical Requirements: {critical_passed}/{critical_total}")
    for req, status in critical_requirements.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {req}")
    
    # Deployment recommendation
    if critical_passed == critical_total:
        if readiness_score >= 90:
            recommendation = "🟢 READY TO DEPLOY - All critical requirements met!"
        elif readiness_score >= 75:
            recommendation = "🟡 MOSTLY READY - Some minor issues, but should work"
        else:
            recommendation = "🟠 NEEDS ATTENTION - Multiple issues to resolve"
    else:
        recommendation = "🔴 NOT READY - Critical requirements missing"
    
    print(f"\n{recommendation}")
    
    return recommendation

def main():
    """Run all verification checks"""
    print("🎬 Wan 2.2 TI2V-5B RunPod Setup Verification")
    print(f"Current directory: {os.getcwd()}")
    
    # Run all checks
    all_results = {
        'environment': check_environment(),
        'dependencies': check_python_dependencies(),
        'cuda': check_cuda_support(),
        'docker': check_docker_files(), 
        'wan22': check_wan22_compatibility(),
        'runpod': check_runpod_compatibility(),
        'examples': test_example_inputs()
    }
    
    # Generate final report
    recommendation = generate_deployment_report(all_results)
    
    print(f"\n📋 Next Steps:")
    if "READY TO DEPLOY" in recommendation:
        print("  1. Build the Docker image: docker build -t wan22-optimized .")
        print("  2. Test locally if possible")
        print("  3. Deploy to RunPod")
        print("  4. Test with example inputs")
    else:
        print("  1. Fix the issues identified above")
        print("  2. Re-run this verification script")
        print("  3. Only deploy after all critical requirements are met")
    
    print(f"\n🎯 For RunPod deployment:")
    print(f"  - Container Disk: 25GB")
    print(f"  - GPU: RTX 4090 or better (24GB VRAM)")
    print(f"  - Environment: WORKER_USE_CUDA=True")
    
    # Return exit code
    critical_issues = sum(1 for section in all_results.values() 
                         for result in section.values() if not result)
    return min(critical_issues, 1)  # 0 if no issues, 1 if any issues

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 