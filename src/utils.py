#!/usr/bin/env python3
"""
Utility functions for Wan2.2 video generation.
"""

import os
import uuid
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import torch
import numpy as np
from diffusers.utils import export_to_video

logger = logging.getLogger(__name__)

def save_video(frames: List[torch.Tensor], output_dir: str, filename: Optional[str] = None, fps: int = 16) -> str:
    """
    Save video frames to MP4 file.
    
    Args:
        frames: List of video frames as tensors
        output_dir: Directory to save the video
        filename: Optional filename (will generate UUID if not provided)
        fps: Frames per second for the output video
        
    Returns:
        str: Path to the saved video file
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = int(time.time())
            unique_id = str(uuid.uuid4())[:8]
            filename = f"wan22_video_{timestamp}_{unique_id}.mp4"
        
        # Ensure filename has .mp4 extension
        if not filename.endswith('.mp4'):
            filename += '.mp4'
            
        video_path = os.path.join(output_dir, filename)
        
        logger.info(f"Saving video to {video_path} with {len(frames)} frames at {fps} FPS")
        
        # Export video using diffusers utility
        export_to_video(frames, video_path, fps=fps)
        
        # Verify the file was created
        if os.path.exists(video_path):
            file_size = os.path.getsize(video_path)
            logger.info(f"Video saved successfully. Size: {file_size / (1024*1024):.2f} MB")
            return video_path
        else:
            raise Exception("Video file was not created")
            
    except Exception as e:
        logger.error(f"Failed to save video: {str(e)}")
        raise

def validate_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and sanitize input parameters.
    
    Args:
        input_data: Raw input data dictionary
        
    Returns:
        dict: Validated and sanitized parameters
    """
    try:
        # Required parameters
        if 'prompt' not in input_data:
            raise ValueError("'prompt' is required in input data")
        
        prompt = str(input_data['prompt']).strip()
        if not prompt:
            raise ValueError("Prompt cannot be empty")
        
        # Optional parameters with defaults
        params = {
            'prompt': prompt,
            'negative_prompt': str(input_data.get('negative_prompt', '')).strip(),
            'num_frames': int(input_data.get('num_frames', 81)),
            'guidance_scale': float(input_data.get('guidance_scale', 5.0)),
            'num_inference_steps': int(input_data.get('num_inference_steps', 40)),
            'fps': int(input_data.get('fps', 16)),
            'seed': input_data.get('seed', None),
            'width': int(input_data.get('width', 720)),
            'height': int(input_data.get('height', 480))
        }
        
        # Validate ranges
        if params['num_frames'] < 1 or params['num_frames'] > 200:
            params['num_frames'] = 81
            logger.warning("num_frames out of range, using default: 81")
            
        if params['guidance_scale'] < 1.0 or params['guidance_scale'] > 20.0:
            params['guidance_scale'] = 5.0
            logger.warning("guidance_scale out of range, using default: 5.0")
            
        if params['num_inference_steps'] < 10 or params['num_inference_steps'] > 100:
            params['num_inference_steps'] = 40
            logger.warning("num_inference_steps out of range, using default: 40")
            
        if params['fps'] < 1 or params['fps'] > 60:
            params['fps'] = 16
            logger.warning("fps out of range, using default: 16")
            
        # Validate resolution
        if params['width'] < 256 or params['width'] > 1920:
            params['width'] = 720
            logger.warning("width out of range, using default: 720")
            
        if params['height'] < 256 or params['height'] > 1080:
            params['height'] = 480
            logger.warning("height out of range, using default: 480")
        
        # Handle seed
        if params['seed'] is not None:
            try:
                params['seed'] = int(params['seed'])
            except ValueError:
                params['seed'] = None
                logger.warning("Invalid seed value, using random seed")
        
        logger.info(f"Validated parameters: {json.dumps(params, indent=2)}")
        return params
        
    except Exception as e:
        logger.error(f"Input validation failed: {str(e)}")
        raise ValueError(f"Invalid input parameters: {str(e)}")

def setup_generator(seed: Optional[int] = None) -> Optional[torch.Generator]:
    """
    Set up random number generator with optional seed.
    
    Args:
        seed: Optional seed for reproducibility
        
    Returns:
        torch.Generator or None
    """
    if seed is not None:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)
        logger.info(f"Using seed: {seed}")
        return generator
    return None

def get_memory_usage() -> Dict[str, float]:
    """
    Get current GPU memory usage.
    
    Returns:
        dict: Memory usage statistics in GB
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return {
            'allocated_gb': round(allocated, 2),
            'reserved_gb': round(reserved, 2),
            'total_gb': round(total, 2),
            'free_gb': round(total - reserved, 2)
        }
    else:
        return {'error': 'CUDA not available'}

def cleanup_temp_files(directory: str, max_age_hours: int = 24) -> int:
    """
    Clean up old temporary files.
    
    Args:
        directory: Directory to clean
        max_age_hours: Maximum age of files to keep
        
    Returns:
        int: Number of files deleted
    """
    try:
        deleted_count = 0
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                file_age = current_time - os.path.getmtime(filepath)
                if file_age > max_age_seconds:
                    try:
                        os.remove(filepath)
                        deleted_count += 1
                        logger.info(f"Deleted old file: {filename}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {filename}: {str(e)}")
        
        return deleted_count
        
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")
        return 0

def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        str: Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def log_system_info():
    """Log system information for debugging."""
    try:
        logger.info("=== System Information ===")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"GPU {i}: {props.name} ({props.total_memory / (1024**3):.1f} GB)")
        
        memory_info = get_memory_usage()
        logger.info(f"Memory usage: {memory_info}")
        logger.info("========================")
        
    except Exception as e:
        logger.error(f"Failed to log system info: {str(e)}")

def create_error_response(error_message: str, error_type: str = "processing_error") -> Dict[str, Any]:
    """
    Create standardized error response.
    
    Args:
        error_message: Error message
        error_type: Type of error
        
    Returns:
        dict: Standardized error response
    """
    return {
        "status": "error",
        "error_type": error_type,
        "message": str(error_message),
        "timestamp": int(time.time())
    }

def create_success_response(video_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create standardized success response.
    
    Args:
        video_path: Path to generated video
        metadata: Additional metadata
        
    Returns:
        dict: Standardized success response
    """
    return {
        "status": "success",
        "video_path": video_path,
        "metadata": metadata,
        "timestamp": int(time.time())
    } 