#!/usr/bin/env python3
"""
Model download script for Wan-AI/Wan2.2-T2V-A14B-Diffusers
Downloads the model to a specified directory if not already present.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_model(model_id: str, local_dir: str, force_download: bool = False) -> bool:
    """
    Download the Wan2.2 model from Hugging Face.
    
    Args:
        model_id: The Hugging Face model ID
        local_dir: Local directory to save the model
        force_download: Whether to force re-download even if model exists
        
    Returns:
        bool: True if download was successful or model already exists
    """
    try:
        local_path = Path(local_dir)
        
        # Check if model already exists and is valid
        if local_path.exists() and not force_download:
            required_files = [
                "model_index.json",
                "scheduler/scheduler_config.json",
                "text_encoder/config.json",
                "unet/config.json",
                "vae/config.json"
            ]
            
            if all((local_path / file).exists() for file in required_files):
                logger.info(f"Model already exists at {local_dir} and appears valid. Skipping download.")
                return True
            else:
                logger.warning("Model directory exists but appears incomplete. Re-downloading...")
        
        # Create directory if it doesn't exist
        local_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading model {model_id} to {local_dir}...")
        logger.info("This may take several minutes for the first download...")
        
        # Download the model
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # Use actual files, not symlinks
            resume_download=True,  # Resume interrupted downloads
        )
        
        logger.info(f"Successfully downloaded model to {local_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download model: {str(e)}")
        return False

def verify_model(model_dir: str) -> bool:
    """
    Verify that the downloaded model is complete and valid.
    
    Args:
        model_dir: Directory containing the model
        
    Returns:
        bool: True if model is valid
    """
    try:
        model_path = Path(model_dir)
        
        # Check for essential files
        required_files = [
            "model_index.json",
            "scheduler/scheduler_config.json", 
            "text_encoder/config.json",
            "unet/config.json",
            "vae/config.json"
        ]
        
        missing_files = []
        for file in required_files:
            if not (model_path / file).exists():
                missing_files.append(file)
        
        if missing_files:
            logger.error(f"Model verification failed. Missing files: {missing_files}")
            return False
            
        logger.info("Model verification successful")
        return True
        
    except Exception as e:
        logger.error(f"Model verification failed: {str(e)}")
        return False

def get_model_size(model_dir: str) -> str:
    """
    Get the approximate size of the downloaded model.
    
    Args:
        model_dir: Directory containing the model
        
    Returns:
        str: Human-readable size string
    """
    try:
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(model_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        
        # Convert to human-readable format
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if total_size < 1024.0:
                return f"{total_size:.1f} {unit}"
            total_size /= 1024.0
        
        return f"{total_size:.1f} PB"
        
    except Exception as e:
        logger.error(f"Failed to calculate model size: {str(e)}")
        return "Unknown"

if __name__ == "__main__":
    # Default configuration
    MODEL_ID = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    DEFAULT_MODEL_DIR = "/runpod-volume/model"
    
    # Use command line arguments if provided
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
    else:
        model_dir = DEFAULT_MODEL_DIR
    
    logger.info(f"Starting model download process...")
    logger.info(f"Model ID: {MODEL_ID}")
    logger.info(f"Target directory: {model_dir}")
    
    # Download the model
    success = download_model(MODEL_ID, model_dir)
    
    if success:
        # Verify the download
        if verify_model(model_dir):
            size = get_model_size(model_dir)
            logger.info(f"Model download and verification complete! Size: {size}")
            sys.exit(0)
        else:
            logger.error("Model verification failed")
            sys.exit(1)
    else:
        logger.error("Model download failed")
        sys.exit(1) 