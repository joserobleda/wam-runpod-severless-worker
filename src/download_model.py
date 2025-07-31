#!/usr/bin/env python3
"""
Download script for CogVideoX-5b model.
This script downloads the model components to the local model_cache directory.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
import torch
from diffusers import CogVideoXTransformer3DModel, AutoencoderKLCogVideoX
from transformers import T5EncoderModel, T5Tokenizer

# Configuration
MODEL_ID = "THUDM/CogVideoX-5b"
MODEL_CACHE_DIR = "model_cache"
HF_CACHE_DIR = "/root/.cache/huggingface"

def download_model():
    """Download the CogVideoX-5b model to local cache."""
    
    print(f"🚀 Starting download of {MODEL_ID}")
    print(f"📁 Cache directory: {MODEL_CACHE_DIR}")
    
    # Create cache directory
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    
    try:
        # Download the complete model
        print("📥 Downloading complete model...")
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=MODEL_CACHE_DIR,
            local_dir_use_symlinks=False,
            cache_dir=HF_CACHE_DIR,
            ignore_patterns=["*.bin"]  # We want safetensors files
        )
        
        print("✅ Model download completed successfully!")
        
        # Verify model components exist
        components_to_check = [
            "transformer",
            "vae", 
            "text_encoder",
            "tokenizer",
            "scheduler"
        ]
        
        for component in components_to_check:
            component_path = Path(MODEL_CACHE_DIR) / component
            if component_path.exists():
                print(f"✅ {component}: Found")
            else:
                print(f"❌ {component}: Missing")
                
        print("🎬 Model ready for video generation!")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading model: {str(e)}")
        return False

def verify_model_loading():
    """Verify that model components can be loaded properly."""
    print("🔍 Verifying model loading...")
    
    try:
        # Test loading each component
        print("Loading transformer...")
        transformer = CogVideoXTransformer3DModel.from_pretrained(
            MODEL_CACHE_DIR, 
            subfolder="transformer",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        print("✅ Transformer loaded successfully")
        
        print("Loading VAE...")
        vae = AutoencoderKLCogVideoX.from_pretrained(
            MODEL_CACHE_DIR, 
            subfolder="vae",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        print("✅ VAE loaded successfully")
        
        print("Loading text encoder...")
        text_encoder = T5EncoderModel.from_pretrained(
            MODEL_CACHE_DIR, 
            subfolder="text_encoder",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        print("✅ Text encoder loaded successfully")
        
        print("Loading tokenizer...")
        tokenizer = T5Tokenizer.from_pretrained(
            MODEL_CACHE_DIR, 
            subfolder="tokenizer"
        )
        print("✅ Tokenizer loaded successfully")
        
        print("🎉 All model components verified successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error verifying model: {str(e)}")
        return False

if __name__ == "__main__":
    print("🤖 CogVideoX-5b Model Downloader")
    print("=" * 50)
    
    # Check if model already exists
    if os.path.exists(MODEL_CACHE_DIR) and os.listdir(MODEL_CACHE_DIR):
        print(f"📁 Model cache directory {MODEL_CACHE_DIR} already exists with content")
        print("🔍 Skipping download, verifying existing model...")
        if verify_model_loading():
            print("✅ Model verification passed!")
            sys.exit(0)
        else:
            print("❌ Model verification failed, re-downloading...")
            import shutil
            shutil.rmtree(MODEL_CACHE_DIR)
    
    # Download model
    if download_model():
        print("🔍 Verifying downloaded model...")
        if verify_model_loading():
            print("🎉 Model download and verification completed successfully!")
            sys.exit(0)
        else:
            print("❌ Model verification failed after download")
            sys.exit(1)
    else:
        print("❌ Model download failed")
        sys.exit(1) 