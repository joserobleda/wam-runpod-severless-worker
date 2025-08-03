#!/usr/bin/env python3
"""
Video generation inference using Wan-AI/Wan2.2-T2V-A14B-Diffusers model.
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
import torch
import gc
from diffusers import WanPipeline
from diffusers.utils import export_to_video

from .utils import setup_generator, get_memory_usage, format_duration

logger = logging.getLogger(__name__)

class Wan22VideoGenerator:
    """Wan2.2 Text-to-Video Generator with optimizations for RunPod deployment."""
    
    def __init__(self, model_dir: str, device: str = "cuda", dtype: str = "bfloat16"):
        """
        Initialize the Wan2.2 video generator.
        
        Args:
            model_dir: Directory containing the downloaded model
            device: Device to run inference on
            dtype: Data type for model weights
        """
        self.model_dir = model_dir
        self.device = device
        self.dtype = getattr(torch, dtype) if hasattr(torch, dtype) else torch.bfloat16
        self.pipeline = None
        self.is_loaded = False
        
        logger.info(f"Initializing Wan22VideoGenerator")
        logger.info(f"Model directory: {model_dir}")
        logger.info(f"Device: {device}")
        logger.info(f"Data type: {self.dtype}")
    
    def load_model(self) -> None:
        """Load the Wan2.2 model pipeline."""
        try:
            start_time = time.time()
            logger.info("Loading Wan2.2 model...")
            
            # Check if CUDA is available
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available. This model requires a GPU.")
            
            # Log initial memory usage
            memory_before = get_memory_usage()
            logger.info(f"GPU memory before loading: {memory_before}")
            
            # Load the main pipeline directly
            logger.info("Loading Wan2.2 pipeline...")
            self.pipeline = WanPipeline.from_pretrained(
                self.model_dir,
                torch_dtype=self.dtype,
                use_safetensors=True
            )
            
            # Move to device
            logger.info(f"Moving pipeline to {self.device}...")
            self.pipeline = self.pipeline.to(self.device)
            
            # Enable memory efficient attention if available
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xformers memory efficient attention")
            except Exception as e:
                logger.warning(f"Could not enable xformers: {e}")
                try:
                    self.pipeline.enable_attention_slicing()
                    logger.info("Enabled attention slicing as fallback")
                except Exception as e2:
                    logger.warning(f"Could not enable attention slicing: {e2}")
            
            # Enable CPU offloading for memory efficiency
            try:
                self.pipeline.enable_model_cpu_offload()
                logger.info("Enabled model CPU offloading")
            except Exception as e:
                logger.warning(f"Could not enable CPU offloading: {e}")
            
            load_time = time.time() - start_time
            memory_after = get_memory_usage()
            
            logger.info(f"Model loaded successfully in {format_duration(load_time)}")
            logger.info(f"GPU memory after loading: {memory_after}")
            
            self.is_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def generate_video(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_frames: int = 81,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 40,
        width: int = 720,
        height: int = 480,
        fps: int = 16,
        seed: Optional[int] = None
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        Generate video from text prompt.
        
        Args:
            prompt: Text description for video generation
            negative_prompt: What to avoid in the video
            num_frames: Number of frames to generate
            guidance_scale: How closely to follow the prompt
            num_inference_steps: Number of denoising steps
            width: Video width
            height: Video height
            fps: Frames per second (for metadata)
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (video_frames, metadata)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            start_time = time.time()
            logger.info(f"Starting video generation...")
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Parameters: frames={num_frames}, guidance={guidance_scale}, steps={num_inference_steps}")
            logger.info(f"Resolution: {width}x{height}, FPS: {fps}")
            
            # Setup generator for reproducibility
            generator = setup_generator(seed)
            
            # Log memory before generation
            memory_before = get_memory_usage()
            logger.info(f"GPU memory before generation: {memory_before}")
            
            # Clear any previous computations
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Generate video
            logger.info("Running diffusion pipeline...")
            
            with torch.cuda.amp.autocast(enabled=True, dtype=self.dtype):
                output = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    num_frames=num_frames,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    width=width,
                    height=height,
                    generator=generator,
                    output_type="pt"  # Return PyTorch tensors
                )
            
            # Extract frames
            frames = output.frames[0] if hasattr(output, 'frames') else output
            
            generation_time = time.time() - start_time
            memory_after = get_memory_usage()
            
            # Create metadata
            metadata = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_frames": len(frames),
                "generation_time_seconds": round(generation_time, 2),
                "generation_time_formatted": format_duration(generation_time),
                "parameters": {
                    "guidance_scale": guidance_scale,
                    "num_inference_steps": num_inference_steps,
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "seed": seed
                },
                "memory_usage": {
                    "before_generation": memory_before,
                    "after_generation": memory_after
                },
                "model_info": {
                    "model_id": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                    "device": self.device,
                    "dtype": str(self.dtype)
                }
            }
            
            logger.info(f"Video generation completed in {format_duration(generation_time)}")
            logger.info(f"Generated {len(frames)} frames")
            logger.info(f"GPU memory after generation: {memory_after}")
            
            return frames, metadata
            
        except Exception as e:
            logger.error(f"Video generation failed: {str(e)}")
            
            # Clean up on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            raise RuntimeError(f"Video generation failed: {str(e)}")
    
    def cleanup(self) -> None:
        """Clean up resources and free memory."""
        try:
            logger.info("Cleaning up resources...")
            
            if self.pipeline is not None:
                # Move pipeline to CPU to free GPU memory
                self.pipeline = self.pipeline.to("cpu")
                del self.pipeline
                self.pipeline = None
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            self.is_loaded = False
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_directory": self.model_dir,
            "device": self.device,
            "dtype": str(self.dtype),
            "is_loaded": self.is_loaded,
            "memory_usage": get_memory_usage() if self.is_loaded else None
        }

# Global model instance (singleton pattern for RunPod serverless)
_model_instance: Optional[Wan22VideoGenerator] = None

def get_model_instance(model_dir: str) -> Wan22VideoGenerator:
    """
    Get or create the global model instance.
    
    Args:
        model_dir: Directory containing the model
        
    Returns:
        Wan22VideoGenerator: The model instance
    """
    global _model_instance
    
    if _model_instance is None:
        _model_instance = Wan22VideoGenerator(model_dir)
        _model_instance.load_model()
    
    return _model_instance

def generate_video(
    prompt: str,
    model_dir: str,
    negative_prompt: str = "",
    num_frames: int = 81,
    guidance_scale: float = 5.0,
    num_inference_steps: int = 40,
    width: int = 720,
    height: int = 480,
    fps: int = 16,
    seed: Optional[int] = None
) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
    """
    Main function to generate video (used by handler).
    
    Args:
        prompt: Text description for video generation
        model_dir: Directory containing the model
        negative_prompt: What to avoid in the video
        num_frames: Number of frames to generate
        guidance_scale: How closely to follow the prompt
        num_inference_steps: Number of denoising steps
        width: Video width
        height: Video height
        fps: Frames per second
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (video_frames, metadata)
    """
    try:
        # Get or create model instance
        model = get_model_instance(model_dir)
        
        # Generate video
        frames, metadata = model.generate_video(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            fps=fps,
            seed=seed
        )
        
        return frames, metadata
        
    except Exception as e:
        logger.error(f"Video generation failed in main function: {str(e)}")
        raise

def cleanup_model():
    """Clean up the global model instance."""
    global _model_instance
    if _model_instance is not None:
        _model_instance.cleanup()
        _model_instance = None 