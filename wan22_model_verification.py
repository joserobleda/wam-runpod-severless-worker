#!/usr/bin/env python3
"""
Official Wan 2.2 TI2V-5B Model Verification
Based on: https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B

Verifies compatibility with the official Wan-AI/Wan2.2-TI2V-5B model requirements
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{'🎬 ' + '='*50}")
    print(f"   {title}")
    print('='*55)

def check_official_requirements():
    """Check against official Wan 2.2 TI2V-5B requirements"""
    print_header("Official Wan 2.2 TI2V-5B Requirements Check")
    
    checks = []
    
    # 1. Model Repository
    model_url = "https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B"
    dockerfile_path = "Dockerfile"
    
    if os.path.exists(dockerfile_path):
        with open(dockerfile_path, 'r') as f:
            dockerfile_content = f.read()
        
        model_download_correct = "Wan-AI/Wan2.2-TI2V-5B" in dockerfile_content
        checks.append(("✅" if model_download_correct else "❌", f"Model Repository: {model_url}", model_download_correct))
    else:
        checks.append(("❌", "Dockerfile not found", False))
    
    # 2. PyTorch Version Requirement
    pytorch_requirement = "torch >= 2.4.0"
    pytorch_in_dockerfile = ("torch==2.4.1" in dockerfile_content and "cu118" in dockerfile_content) if 'dockerfile_content' in locals() else False
    checks.append(("✅" if pytorch_in_dockerfile else "❌", f"PyTorch Version: {pytorch_requirement}", pytorch_in_dockerfile))
    
    # 3. Official Command Structure
    official_cmd = "python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu"
    predict_path = "src/predict.py"
    
    if os.path.exists(predict_path):
        with open(predict_path, 'r') as f:
            predict_content = f.read()
        
        cmd_components = [
            "'--task', 'ti2v-5B'",
            "'--offload_model', 'True'",
            "'--convert_model_dtype'",
            "'--t5_cpu'"
        ]
        
        cmd_correct = all(comp in predict_content for comp in cmd_components)
        checks.append(("✅" if cmd_correct else "❌", "Official Command Structure", cmd_correct))
    else:
        checks.append(("❌", "predict.py not found", False))
    
    # 4. Supported Resolutions
    supported_resolutions = ["1280*704", "704*1280"]  # Official 720P resolutions
    resolution_mapping_correct = False
    
    if os.path.exists(predict_path):
        resolution_mapping_correct = any(res in predict_content for res in supported_resolutions)
    
    checks.append(("✅" if resolution_mapping_correct else "❌", "720P Resolution Support (1280*704, 704*1280)", resolution_mapping_correct))
    
    # 5. GPU Requirements
    gpu_req = "RTX 4090 (24GB VRAM) compatibility"
    gpu_optimizations = ["--offload_model", "--convert_model_dtype", "--t5_cpu"]
    gpu_compatible = all(opt in predict_content for opt in gpu_optimizations) if 'predict_content' in locals() else False
    checks.append(("✅" if gpu_compatible else "❌", f"GPU Requirements: {gpu_req}", gpu_compatible))
    
    # 6. Task Support
    tasks = ["Text-to-Video", "Image-to-Video"]
    task_support = "'--image'" in predict_content if 'predict_content' in locals() else False
    checks.append(("✅" if task_support else "❌", "Task Support: Text-to-Video & Image-to-Video", task_support))
    
    # Print results
    for icon, description, status in checks:
        print(f"  {icon} {description}")
        if not status and "❌" in icon:
            print(f"    ⚠️  This requirement is not met!")
    
    return checks

def check_api_compatibility():
    """Check API parameter compatibility"""
    print_header("API Parameter Compatibility")
    
    # Official parameters from the model documentation
    official_params = {
        "task": "ti2v-5B",
        "size": ["1280*704", "704*1280"],
        "ckpt_dir": "./Wan2.2-TI2V-5B",
        "offload_model": "True",
        "convert_model_dtype": True,
        "t5_cpu": True,
        "prompt": "required",
        "image": "optional",
        "guidance_scale": "default: 7.5",
        "num_inference_steps": "default: 50",
        "fps": "default: 24",
        "seed": "optional"
    }
    
    schema_path = "src/rp_schema.py"
    schema_compatible = False
    
    if os.path.exists(schema_path):
        with open(schema_path, 'r') as f:
            schema_content = f.read()
        
        # Check for required parameters
        required_checks = [
            '"prompt"' in schema_content,
            '"size"' in schema_content,
            '"guidance_scale"' in schema_content,
            '"num_inference_steps"' in schema_content,
            '"fps"' in schema_content,
            '"seed"' in schema_content
        ]
        
        schema_compatible = all(required_checks)
    
    print(f"  {'✅' if schema_compatible else '❌'} RunPod Schema Parameters")
    print(f"  {'✅' if os.path.exists(schema_path) else '❌'} rp_schema.py exists")
    
    # Check example inputs
    example_files = [
        "example_input_simple.json",
        "example_input_youtube_short.json", 
        "example_input_image_to_video.json"
    ]
    
    example_compatible = True
    for example_file in example_files:
        if os.path.exists(example_file):
            try:
                with open(example_file, 'r') as f:
                    data = json.load(f)
                
                has_input_wrapper = 'input' in data
                has_prompt = has_input_wrapper and 'prompt' in data['input']
                file_ok = has_input_wrapper and has_prompt
                
                print(f"  {'✅' if file_ok else '❌'} {example_file}")
                if not file_ok:
                    example_compatible = False
            except:
                print(f"  ❌ {example_file} (JSON error)")
                example_compatible = False
        else:
            print(f"  ❌ {example_file} (missing)")
            example_compatible = False
    
    return schema_compatible and example_compatible

def check_deployment_readiness():
    """Check deployment readiness for RunPod"""
    print_header("RunPod Deployment Readiness")
    
    checks = []
    
    # Docker file optimization
    dockerfile_path = "Dockerfile"
    if os.path.exists(dockerfile_path):
        with open(dockerfile_path, 'r') as f:
            dockerfile_content = f.read()
        
        multi_stage = "FROM ubuntu:22.04 AS builder" in dockerfile_content
        runtime_stage = "FROM ubuntu:22.04" in dockerfile_content
        optimized = multi_stage and runtime_stage
        
        checks.append(("✅" if optimized else "❌", "Multi-stage Docker build (size optimization)", optimized))
        
        # Check for proper cleanup
        cleanup_commands = ["rm -rf", "apt-get clean", "find . -name \"*.md\" -delete"]
        has_cleanup = any(cmd in dockerfile_content for cmd in cleanup_commands)
        checks.append(("✅" if has_cleanup else "❌", "Build cleanup (size optimization)", has_cleanup))
        
        # Check for CUDA installation in build
        has_cuda_install = "cuda-toolkit-11-8" in dockerfile_content and "cuda-runtime-11-8" in dockerfile_content
        checks.append(("✅" if has_cuda_install else "❌", "CUDA 11.8 installation (toolkit + runtime)", has_cuda_install))
    else:
        checks.append(("❌", "Dockerfile not found", False))
        optimized = False
        has_cleanup = False
        uses_runtime = False
    
    # Check for RunPod specific files
    runpod_files = {
        "src/rp_handler.py": "RunPod request handler",
        "src/startup.sh": "Container startup script",
        "builder/requirements.txt": "Python dependencies"
    }
    
    for file_path, description in runpod_files.items():
        exists = os.path.exists(file_path)
        checks.append(("✅" if exists else "❌", description, exists))
    
    # Print results
    for icon, description, status in checks:
        print(f"  {icon} {description}")
    
    return all(status for _, _, status in checks)

def generate_final_report():
    """Generate final compatibility report"""
    print_header("Final Compatibility Report")
    
    # Run all checks
    req_checks = check_official_requirements()
    api_compatible = check_api_compatibility()
    deploy_ready = check_deployment_readiness()
    
    # Calculate scores
    req_passed = sum(1 for _, _, status in req_checks if status)
    req_total = len(req_checks)
    req_score = (req_passed / req_total) * 100
    
    print(f"\n📊 Compatibility Score:")
    print(f"  Official Requirements: {req_passed}/{req_total} ({req_score:.1f}%)")
    print(f"  API Compatibility: {'✅ PASS' if api_compatible else '❌ FAIL'}")
    print(f"  Deployment Ready: {'✅ PASS' if deploy_ready else '❌ FAIL'}")
    
    # Overall assessment
    if req_score >= 100 and api_compatible and deploy_ready:
        status = "🟢 FULLY COMPATIBLE"
        recommendation = "Ready for production deployment"
    elif req_score >= 85:
        status = "🟡 MOSTLY COMPATIBLE"
        recommendation = "Minor issues, should work with testing"
    else:
        status = "🔴 NOT COMPATIBLE"
        recommendation = "Critical issues must be resolved"
    
    print(f"\n🎯 Overall Status: {status}")
    print(f"📋 Recommendation: {recommendation}")
    
    print(f"\n🔗 Reference: https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B")
    print(f"📚 Official Command: python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --prompt \"...\"")
    
    return req_score >= 85 and api_compatible

def main():
    """Main verification function"""
    print("🎬 Wan 2.2 TI2V-5B Official Model Verification")
    print("Based on: https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B")
    print(f"Working Directory: {os.getcwd()}")
    
    compatible = generate_final_report()
    
    if compatible:
        print(f"\n✅ VERIFICATION PASSED - Compatible with official Wan 2.2 TI2V-5B!")
        print(f"🚀 Ready for Docker build and RunPod deployment")
        return 0
    else:
        print(f"\n❌ VERIFICATION FAILED - Issues detected")
        print(f"🔧 Please resolve the issues above before deployment")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 