#!/usr/bin/env python3
"""
RunPod serverless handler for Wan-AI/Wan2.2-T2V-A14B-Diffusers model.
This is the main entry point for the serverless deployment.
"""

import os
import sys
import time
import logging
import signal
import atexit
from pathlib import Path
from typing import Dict, Any

# Add builder path for model downloading
sys.path.append('/builder')
sys.path.append('/src')

import runpod
from builder.download_model import download_model, verify_model
from src.predict import generate_video, cleanup_model
from src.utils import (
    validate_input, save_video, create_error_response, 
    create_success_response, log_system_info, get_memory_usage,
    cleanup_temp_files, format_duration
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_ID = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
MODEL_DIR = "/runpod-volume/model"
OUTPUT_DIR = "/runpod-volume/outputs"
TEMP_CLEANUP_HOURS = 6  # Clean up temp files older than 6 hours

class Wan22Handler:
    """Main handler class for Wan2.2 video generation."""
    
    def __init__(self):
        """Initialize the handler."""
        self.model_loaded = False
        self.startup_time = time.time()
        
        logger.info("Initializing Wan22Handler...")
        log_system_info()
        
        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Setup cleanup on exit
        atexit.register(self.cleanup_on_exit)
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, cleaning up...")
        self.cleanup_on_exit()
        sys.exit(0)
    
    def cleanup_on_exit(self):
        """Cleanup resources on exit."""
        try:
            logger.info("Performing cleanup on exit...")
            cleanup_model()
            
            # Clean up old temporary files
            cleaned_files = cleanup_temp_files(OUTPUT_DIR, TEMP_CLEANUP_HOURS)
            if cleaned_files > 0:
                logger.info(f"Cleaned up {cleaned_files} old files")
            
            logger.info("Exit cleanup completed")
        except Exception as e:
            logger.error(f"Error during exit cleanup: {e}")
    
    def ensure_model_ready(self) -> bool:
        """Ensure the model is downloaded and ready."""
        try:
            if not os.path.exists(MODEL_DIR):
                logger.info(f"Model directory not found at {MODEL_DIR}. Downloading model...")
                success = download_model(MODEL_ID, MODEL_DIR)
                if not success:
                    raise RuntimeError("Failed to download model")
            
            # Verify model integrity
            if not verify_model(MODEL_DIR):
                logger.warning("Model verification failed. Re-downloading...")
                success = download_model(MODEL_ID, MODEL_DIR, force_download=True)
                if not success:
                    raise RuntimeError("Failed to re-download model after verification failure")
            
            logger.info("Model is ready for inference")
            return True
            
        except Exception as e:
            logger.error(f"Failed to ensure model ready: {str(e)}")
            return False
    
    def process_request(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a video generation request.
        
        Args:
            event: RunPod event data
            
        Returns:
            dict: Response data
        """
        request_start_time = time.time()
        
        try:
            logger.info("=" * 50)
            logger.info("Processing new video generation request")
            logger.info(f"Request data: {event}")
            
            # Validate input
            if 'input' not in event:
                return create_error_response("Missing 'input' in request", "validation_error")
            
            try:
                params = validate_input(event['input'])
            except ValueError as e:
                return create_error_response(str(e), "validation_error")
            
            # Ensure model is ready
            if not self.ensure_model_ready():
                return create_error_response("Model setup failed", "model_error")
            
            # Log memory before generation
            memory_before = get_memory_usage()
            logger.info(f"Memory before generation: {memory_before}")
            
            # Generate video
            logger.info("Starting video generation...")
            generation_start_time = time.time()
            
            frames, metadata = generate_video(
                prompt=params['prompt'],
                model_dir=MODEL_DIR,
                negative_prompt=params['negative_prompt'],
                num_frames=params['num_frames'],
                guidance_scale=params['guidance_scale'],
                num_inference_steps=params['num_inference_steps'],
                width=params['width'],
                height=params['height'],
                fps=params['fps'],
                seed=params['seed']
            )
            
            generation_time = time.time() - generation_start_time
            logger.info(f"Video generation completed in {format_duration(generation_time)}")
            
            # Save video
            logger.info("Saving video...")
            save_start_time = time.time()
            
            video_path = save_video(
                frames=frames,
                output_dir=OUTPUT_DIR,
                fps=params['fps']
            )
            
            save_time = time.time() - save_start_time
            logger.info(f"Video saved in {format_duration(save_time)}")
            
            # Create response metadata
            total_time = time.time() - request_start_time
            memory_after = get_memory_usage()
            
            response_metadata = {
                **metadata,
                "timing": {
                    "total_request_time_seconds": round(total_time, 2),
                    "generation_time_seconds": round(generation_time, 2),
                    "save_time_seconds": round(save_time, 2),
                    "total_request_time_formatted": format_duration(total_time)
                },
                "memory_usage_after": memory_after,
                "video_info": {
                    "path": video_path,
                    "filename": os.path.basename(video_path),
                    "size_bytes": os.path.getsize(video_path) if os.path.exists(video_path) else 0
                }
            }
            
            logger.info(f"Request completed successfully in {format_duration(total_time)}")
            logger.info(f"Video saved to: {video_path}")
            logger.info("=" * 50)
            
            return create_success_response(video_path, response_metadata)
            
        except Exception as e:
            total_time = time.time() - request_start_time
            logger.error(f"Request failed after {format_duration(total_time)}: {str(e)}")
            logger.error("=" * 50)
            
            return create_error_response(str(e), "processing_error")

# Create global handler instance
handler_instance = Wan22Handler()

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main RunPod handler function.
    
    Args:
        event: RunPod event data
        
    Returns:
        dict: Response data
    """
    return handler_instance.process_request(event)

def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for monitoring.
    
    Returns:
        dict: Health status information
    """
    try:
        uptime = time.time() - handler_instance.startup_time
        memory_usage = get_memory_usage()
        
        return {
            "status": "healthy",
            "uptime_seconds": round(uptime, 2),
            "uptime_formatted": format_duration(uptime),
            "model_ready": os.path.exists(MODEL_DIR) and verify_model(MODEL_DIR),
            "model_directory": MODEL_DIR,
            "output_directory": OUTPUT_DIR,
            "memory_usage": memory_usage,
            "timestamp": int(time.time())
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": int(time.time())
        }

if __name__ == "__main__":
    logger.info("Starting Wan2.2 RunPod Serverless Worker...")
    logger.info(f"Model ID: {MODEL_ID}")
    logger.info(f"Model Directory: {MODEL_DIR}")
    logger.info(f"Output Directory: {OUTPUT_DIR}")
    
    try:
        # Perform initial setup
        logger.info("Performing initial setup...")
        
        # Pre-download model if not present
        if not handler_instance.ensure_model_ready():
            logger.error("Initial model setup failed!")
            sys.exit(1)
        
        logger.info("Initial setup completed successfully")
        logger.info("Worker is ready to accept requests")
        
        # Start the RunPod serverless worker
        runpod.serverless.start({
            "handler": handler,
            "return_aggregate_stream": True,
        })
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
        handler_instance.cleanup_on_exit()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start worker: {str(e)}")
        handler_instance.cleanup_on_exit()
        sys.exit(1) 