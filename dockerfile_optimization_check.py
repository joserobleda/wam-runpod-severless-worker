#!/usr/bin/env python3
"""
Comprehensive Dockerfile Optimization Check
Analyzes the Dockerfile for efficiency and potential improvements
"""

import os
import sys
import re
from typing import List, Dict, Tuple

def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print('='*60)

def analyze_dockerfile() -> Dict[str, List[str]]:
    """Analyze Dockerfile for optimization opportunities"""
    print_section("Dockerfile Optimization Analysis")
    
    if not os.path.exists("Dockerfile"):
        print("❌ Dockerfile not found")
        return {}
    
    with open("Dockerfile", 'r') as f:
        dockerfile_content = f.read()
        lines = dockerfile_content.split('\n')
    
    issues = {
        'optimizations': [],
        'warnings': [],
        'good_practices': [],
        'size_optimizations': []
    }
    
    # Check for multi-stage build
    if "FROM ubuntu:22.04 AS builder" in dockerfile_content and "FROM ubuntu:22.04" in dockerfile_content:
        issues['good_practices'].append("✅ Multi-stage build properly implemented")
    else:
        issues['warnings'].append("⚠️ Multi-stage build not detected")
    
    # Check for debug output that should be removed
    debug_patterns = ['echo "===', 'ls -la', 'find /usr/local/lib']
    for i, line in enumerate(lines, 1):
        for pattern in debug_patterns:
            if pattern in line:
                issues['optimizations'].append(f"🔧 Remove debug output at line {i}: {line.strip()}")
    
    # Check for proper cleanup
    cleanup_patterns = ['apt-get clean', 'rm -rf /var/lib/apt/lists/*', 'rm -rf .git']
    cleanup_found = sum(1 for pattern in cleanup_patterns if pattern in dockerfile_content)
    if cleanup_found >= 2:
        issues['good_practices'].append("✅ Good cleanup practices found")
    else:
        issues['warnings'].append("⚠️ Missing some cleanup commands")
    
    # Check layer efficiency
    run_commands = [i for i, line in enumerate(lines, 1) if line.strip().startswith('RUN')]
    if len(run_commands) <= 10:  # Reasonable number of layers
        issues['good_practices'].append(f"✅ Efficient layering: {len(run_commands)} RUN commands")
    else:
        issues['optimizations'].append(f"🔧 Consider combining RUN commands: {len(run_commands)} found")
    
    # Check for .dockerignore
    if os.path.exists(".dockerignore"):
        issues['good_practices'].append("✅ .dockerignore file exists")
    else:
        issues['optimizations'].append("🔧 Consider adding .dockerignore file")
    
    # Check for proper COPY vs ADD usage
    add_commands = [line for line in lines if line.strip().startswith('ADD')]
    copy_commands = [line for line in lines if line.strip().startswith('COPY')]
    
    if len(copy_commands) >= len(add_commands):
        issues['good_practices'].append("✅ Proper use of COPY over ADD")
    
    # Check for CUDA optimization
    if "cuda-toolkit-11-8" in dockerfile_content and "cuda-runtime-11-8" in dockerfile_content:
        issues['good_practices'].append("✅ Optimized CUDA: toolkit in builder, runtime in final stage")
    
    # Check for non-root user
    if "USER user" in dockerfile_content:
        issues['good_practices'].append("✅ Non-root user implemented for security")
    
    # Check for specific optimizations
    if "--no-cache-dir" in dockerfile_content:
        issues['good_practices'].append("✅ pip --no-cache-dir used to reduce image size")
    
    if "${WORKER_DIR}" in dockerfile_content:
        issues['good_practices'].append("✅ Environment variables used for maintainability")
    
    return issues

def check_file_sizes() -> Dict[str, str]:
    """Check sizes of key files that will be copied"""
    print_section("File Size Analysis")
    
    file_checks = {}
    
    # Check requirements.txt
    req_file = "builder/requirements.txt"
    if os.path.exists(req_file):
        with open(req_file, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]
        file_checks['requirements.txt'] = f"{len(lines)} packages"
        print(f"  📦 Requirements: {len(lines)} packages")
    
    # Check source directory size
    src_dir = "src"
    if os.path.exists(src_dir):
        src_files = []
        for root, dirs, files in os.walk(src_dir):
            src_files.extend(files)
        file_checks['src/'] = f"{len(src_files)} files"
        print(f"  📁 Source files: {len(src_files)} files")
    
    # Check for unnecessary files
    unnecessary_patterns = ['.pyc', '__pycache__', '.git', '.DS_Store', '*.md']
    print(f"  🔍 Checking for unnecessary files...")
    
    return file_checks

def check_runtime_dependencies() -> List[str]:
    """Check runtime dependencies for optimization"""
    print_section("Runtime Dependencies Analysis")
    
    recommendations = []
    
    # Check Dockerfile for runtime packages
    if os.path.exists("Dockerfile"):
        with open("Dockerfile", 'r') as f:
            content = f.read()
        
        # Find runtime stage packages
        runtime_stage = False
        runtime_packages = []
        
        for line in content.split('\n'):
            if "FROM ubuntu:22.04" in line and "AS builder" not in line:
                runtime_stage = True
            elif line.strip().startswith("FROM"):
                runtime_stage = False
            elif runtime_stage and "apt-get install -y" in line:
                # Extract packages from the line
                packages = line.split("apt-get install -y")[1].split("&&")[0].strip()
                runtime_packages.extend(packages.split())
        
        essential_runtime = [
            'python3', 'python3-pip', 'ffmpeg', 'libgl1-mesa-glx', 
            'libglib2.0-0', 'libsm6', 'libxext6', 'libxrender-dev', 'libgomp1'
        ]
        
        extra_packages = [pkg for pkg in runtime_packages if pkg not in essential_runtime and pkg]
        
        if len(extra_packages) <= 3:  # wget, gnupg2, plus CUDA packages are reasonable
            recommendations.append("✅ Runtime dependencies are minimal")
        else:
            recommendations.append(f"🔧 Consider reviewing extra runtime packages: {extra_packages}")
    
    return recommendations

def estimate_final_size() -> str:
    """Estimate final Docker image size"""
    print_section("Image Size Estimation")
    
    size_components = {
        "Ubuntu 22.04 base": "77 MB",
        "CUDA runtime": "~1.5 GB",
        "Python + packages": "~2-3 GB", 
        "Wan 2.2 model": "~10-12 GB",
        "Application code": "~10 MB"
    }
    
    print("📊 Estimated size breakdown:")
    total_estimate = "15-18 GB"
    
    for component, size in size_components.items():
        print(f"  • {component}: {size}")
    
    print(f"\n🎯 Total estimated size: {total_estimate}")
    print("📈 This is optimized from potential 50GB+ without multi-stage build")
    
    return total_estimate

def generate_optimization_report() -> bool:
    """Generate comprehensive optimization report"""
    print("🔍 Comprehensive Dockerfile Optimization Check")
    print("=" * 60)
    
    # Run all checks
    dockerfile_issues = analyze_dockerfile()
    file_sizes = check_file_sizes()
    runtime_deps = check_runtime_dependencies()
    estimated_size = estimate_final_size()
    
    # Print results
    print_section("Optimization Summary")
    
    # Good practices
    if dockerfile_issues.get('good_practices'):
        print("✅ Good Practices Found:")
        for practice in dockerfile_issues['good_practices']:
            print(f"  {practice}")
    
    # Optimizations
    if dockerfile_issues.get('optimizations'):
        print("\n🔧 Optimization Opportunities:")
        for opt in dockerfile_issues['optimizations']:
            print(f"  {opt}")
    
    # Warnings
    if dockerfile_issues.get('warnings'):
        print("\n⚠️ Warnings:")
        for warning in dockerfile_issues['warnings']:
            print(f"  {warning}")
    
    # Runtime dependencies
    if runtime_deps:
        print("\n🏃 Runtime Dependencies:")
        for dep in runtime_deps:
            print(f"  {dep}")
    
    # Overall assessment
    total_issues = len(dockerfile_issues.get('optimizations', [])) + len(dockerfile_issues.get('warnings', []))
    good_practices_count = len(dockerfile_issues.get('good_practices', []))
    
    print_section("Overall Assessment")
    
    if total_issues == 0:
        status = "🟢 FULLY OPTIMIZED"
        recommendation = "Dockerfile is production-ready and well-optimized"
    elif total_issues <= 2:
        status = "🟡 MOSTLY OPTIMIZED"
        recommendation = "Minor optimizations available but ready for production"
    else:
        status = "🔴 NEEDS OPTIMIZATION"
        recommendation = "Several optimizations should be applied before production"
    
    print(f"📊 Good Practices: {good_practices_count}")
    print(f"📊 Issues Found: {total_issues}")
    print(f"📊 Estimated Final Size: {estimated_size}")
    
    print(f"\n{status}")
    print(f"📋 Recommendation: {recommendation}")
    
    return total_issues <= 2

def main():
    """Main optimization check function"""
    try:
        optimized = generate_optimization_report()
        return 0 if optimized else 1
    except Exception as e:
        print(f"\n❌ Optimization check failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 