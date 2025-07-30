"""
Simplified Wan 2.2 TI2V-5B using Diffusers with Full Parameter Support
Based on official Hugging Face documentation
"""

import torch
import numpy as np
from diffusers import WanPipeline, WanTransformer3DModel, UniPCMultistepScheduler, AutoencoderKL
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
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
        self.dtype = torch.float32  # Force float32 for numerical stability
    
    def setup(self):
        """Initialize the Diffusers pipeline exactly like official documentation"""
        logger.info("🚀 Setting up Wan 2.2 TI2V-5B with Diffusers...")
        
        # Use the DIFFUSERS-COMPATIBLE version of the model
        model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
        
        try:
            logger.info(f"🔄 Loading WanPipeline from {model_id}...")
            
            # Explicitly load the VAE with `ignore_mismatched_sizes` to handle
            # architecture conflicts between the checkpoint and the pipeline config.
            logger.info("📦 Loading VAE separately to handle size mismatches...")
            vae = AutoencoderKLWan.from_pretrained(
                model_id,
                subfolder="vae",
                torch_dtype=self.dtype,
                low_cpu_mem_usage=False,  # Load VAE fully into memory to avoid meta tensor issues
                trust_remote_code=True,
                ignore_mismatched_sizes=True,
            )
            logger.info("✅ VAE loaded successfully.")

            # Load WanPipeline, providing the custom-loaded VAE
            logger.info("📦 Loading WanPipeline with the custom VAE...")
            self.pipe = WanPipeline.from_pretrained(
                model_id,
                vae=vae,
                torch_dtype=self.dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            
            # Move to device (explicit approach as per docs)
            logger.info(f"🔧 Moving pipeline to {self.device}...")
            self.pipe.to(self.device)
            
            logger.info(f"✅ WanPipeline loaded successfully with dtype {self.dtype}")
            logger.info(f"✅ Pipeline moved to {self.device}")
            logger.info("🚀 Ready for video generation!")
            
        except Exception as e:
            logger.error(f"❌ Failed to load WanPipeline: {e}")
            logger.error(f"❌ Error type: {type(e).__name__}")
            
            # Check if it's a specific component issue
            error_msg = str(e).lower()
            if 'autoencoder' in error_msg or 'vae' in error_msg:
                logger.info("💡 Issue with VAE loading - this is critical for Wan2.2")
            elif 'wantransformer3d' in error_msg:
                logger.info("💡 Issue with transformer model loading")
            elif 'not found' in error_msg:
                logger.info("💡 Model components not found - check model ID")
            
            logger.info("🔗 Check the model page: https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers")
            logger.info("🔗 Official Wan2.2 repo: https://github.com/Wan-Video/Wan2.2")
            
            raise RuntimeError(f"WanPipeline loading failed: {e}. Check model compatibility.")
    
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
            
            # Default negative prompt (English version for better compatibility)
            if negative_prompt is None:
                negative_prompt = "blurry, low quality, static, poorly drawn, deformed, disfigured, extra limbs, fused fingers, malformed hands, ugly, distorted, overexposed, underexposed, saturated colors, artifacts, compression artifacts, pixelated, grainy, watermark, signature, text, subtitles, still image, frozen, cluttered background, multiple people, crowd, walking backwards, unnatural movement"
            
            # Load image if provided (Image-to-Video)
            image = None
            if image_path and os.path.exists(image_path):
                image = load_image(image_path)
                logger.info(f"Loaded image for I2V: {image_path}")
            
            logger.info("Starting generation...")
            
            # Build pipeline parameters
            pipeline_params = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
            }
            
            # Add image only for Image-to-Video (when image is provided)
            if image is not None:
                pipeline_params["image"] = image
                logger.info("Using I2V mode with provided image")
            else:
                logger.info("Using T2V mode (text-to-video)")
                
            # Add generator for seed if provided
            if seed is not None:
                pipeline_params["generator"] = torch.Generator(device=self.device).manual_seed(seed)
            
            # Generate video with dynamic parameters
            output = self.pipe(**pipeline_params).frames[0]
            
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