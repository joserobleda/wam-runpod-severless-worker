"""
Simplified Wan 2.2 TI2V-5B using Diffusers with Full Parameter Support
Based on official Hugging Face documentation
"""

import torch
import numpy as np
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video, load_image
from typing import Optional
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplifiedWanPredictor:
    """Simplified Wan 2.2 predictor using Diffusers - supports ALL parameters"""
    
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    def setup(self):
        """Initialize the Diffusers pipeline"""
        logger.info("🚀 Setting up Wan 2.2 TI2V-5B with Diffusers...")
        
        # Use the official Wan 2.2 model
        model_id = "Wan-AI/Wan2.2-TI2V-5B"
        
        try:
            # Try to load with DiffusionPipeline (if diffusers support is available)
            self.pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                trust_remote_code=True,  # Important for custom models
                device_map="auto"
            )
            
            # Move to device if not using device_map="auto"
            if not hasattr(self.pipe, '_device_map'):
                self.pipe.to(self.device)
            
            logger.info(f"✓ Pipeline loaded with dtype {self.dtype}")
            
        except Exception as e:
            logger.error(f"Failed to load pipeline with diffusers: {e}")
            logger.info("Note: Wan 2.2 diffusers integration may not be available yet")
            logger.info("Check the model page: https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B")
            logger.info("You may need to wait for official diffusers support or use the subprocess approach")
            raise RuntimeError(f"Diffusers pipeline loading failed: {e}. Wan 2.2 may not support diffusers yet.")
    
    def predict(self, 
                prompt: str,
                image_path: Optional[str] = None,
                size: str = "1280*704",  # Wan format: width*height
                num_frames: int = 121,   # Wan 2.2 default (5 seconds at 24fps)
                guidance_scale: float = 5.0,  # Wan 2.2 optimized default
                num_inference_steps: int = 50,
                seed: Optional[int] = None,
                fps: int = 24,
                negative_prompt: Optional[str] = None,
                use_prompt_extend: bool = False) -> str:
        """
        Generate video with FULL parameter support using Diffusers
        
        Args:
            prompt: Text description of the video
            image_path: Optional image for I2V generation
            size: Video resolution in "width*height" format
            num_frames: Number of frames to generate (121 = 5s at 24fps)
            guidance_scale: How closely to follow the prompt (1.0-20.0)
            num_inference_steps: Quality vs speed tradeoff (20-100)
            seed: Random seed for reproducibility
            fps: Frames per second for output video
            negative_prompt: What to avoid in the generation
            use_prompt_extend: Whether to extend short prompts
        """
        
        try:
            logger.info(f"🎬 Generating video with Diffusers...")
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Size: {size}, Frames: {num_frames}, Steps: {num_inference_steps}")
            logger.info(f"Guidance: {guidance_scale}, FPS: {fps}, Seed: {seed}")
            
            # Set seed for reproducibility
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
            
            # Extend prompt if requested
            if use_prompt_extend and len(prompt.strip()) < 50:
                prompt = f"{prompt.strip()}, cinematic quality, high detail, professional lighting, smooth motion, 720P resolution"
                logger.info(f"Extended prompt: {prompt}")
            
            # Parse size (convert from "1280*704" to width, height)
            if '*' in size:
                width, height = map(int, size.split('*'))
            elif 'x' in size:
                width, height = map(int, size.split('x'))
            else:
                # Default to 720p landscape
                width, height = 1280, 704
            
            # Default negative prompt (official Wan 2.2 recommendation)
            if negative_prompt is None:
                negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
            
            # Load image if provided (Image-to-Video)
            image = None
            if image_path and os.path.exists(image_path):
                image = load_image(image_path)
                logger.info(f"Loaded image for I2V: {image_path}")
            
            logger.info("Starting generation...")
            
            # Generate video with FULL parameter support
            output = self.pipe(
                prompt=prompt,
                image=image,  # None for T2V, loaded image for I2V
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                # Additional parameters supported by Diffusers:
                generator=torch.Generator(device=self.device).manual_seed(seed) if seed else None,
                # eta=0.0,  # DDIM eta parameter
                # callback=None,  # Optional callback function
                # callback_steps=1,  # How often to call callback
            ).frames[0]
            
            # Export to video file
            output_path = f"/tmp/video_generation/wan22_diffusers_{os.getpid()}_{seed or 'random'}.mp4"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            export_to_video(output, output_path, fps=fps)
            
            # Verify output
            if not os.path.exists(output_path):
                raise RuntimeError("Video export failed")
            
            file_size = os.path.getsize(output_path)
            logger.info(f"✓ Video generated: {output_path} ({file_size / (1024*1024):.2f} MB)")
            
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Video generation failed: {e}")
            raise RuntimeError(f"Video generation failed: {e}")


# RunPod integration
MODEL = None

def setup():
    """Setup function for RunPod"""
    global MODEL
    if MODEL is None:
        model_dir = os.environ.get('WORKER_MODEL_DIR', '/app/model')
        MODEL = SimplifiedWanPredictor(model_dir)
        MODEL.setup()
    return MODEL

def predict(prompt: str, **kwargs) -> str:
    """Predict function for RunPod with full parameter support"""
    global MODEL
    if MODEL is None:
        MODEL = setup()
    return MODEL.predict(prompt, **kwargs)