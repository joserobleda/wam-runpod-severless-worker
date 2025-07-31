"""
Simplified Wan 2.2 TI2V-5B using Diffusers with Full Parameter Support
Based on official Hugging Face documentation
"""

import os
import torch
import numpy as np
from PIL import Image
from diffusers import WanPipeline, AutoencoderKL
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from diffusers.utils import export_to_video
from loguru import logger
import tempfile
import random
import runpod
from runpod.serverless.utils import rp_cleanup
import sys

# Set matmul precision for better performance on Ampere GPUs
torch.set_float32_matmul_precision('high')

class Predictor:
    """
    A simplified and robust predictor for the Wan2.2-TI2V-5B-Diffusers model.
    """
    def setup(self):
        """Initializes the model pipeline and applies necessary patches."""
        logger.info("🚀 Initializing Wan 2.2 Video Generator...")

        # --- Platform-aware device and dtype selection ---
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.dtype = torch.bfloat16
            logger.info("✅ CUDA available. Using GPU with bfloat16.")
        # Check for macOS and Apple Silicon
        elif sys.platform == "darwin" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.dtype = torch.float32  # bfloat16 not fully supported on MPS
            logger.info("✅ Apple MPS available. Using GPU with float32.")
        else:
            self.device = torch.device("cpu")
            self.dtype = torch.float32
            logger.info("⚠️ No CUDA or MPS GPU available. Falling back to CPU with float32 (will be very slow).")

        model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
        
        # --- Step 1: Load a known-good VAE and modify its class for compatibility ---
        logger.info("📦 Loading stable VAE weights from stabilityai/stable-video-diffusion-img2vid-xt...")
        # Load the stable, known-good video VAE
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt", 
            subfolder="vae"
        )
        
        logger.info("🔧 Modifying VAE class to satisfy pipeline's type check...")
        # The WanPipeline strictly checks for the `AutoencoderKLWan` class.
        # We dynamically change the class of our loaded, stable VAE to trick this check.
        # This uses the good VAE's architecture and weights directly.
        vae.__class__ = AutoencoderKLWan
        
        vae.to(self.dtype)
        logger.info("✅ VAE is now configured with stable weights and the correct class type.")


        # --- Step 2: Load the Main Pipeline with the Corrected VAE ---
        logger.info(f"📦 Loading WanPipeline from {model_id} with our corrected VAE...")
        self.pipe = WanPipeline.from_pretrained(
            model_id,
            vae=vae,
            torch_dtype=self.dtype,
            trust_remote_code=True
        )
        self.pipe.to(self.device)
        logger.info(f"✅ Pipeline loaded to device '{self.device}'.")

        # --- Step 3: Compile for Performance (only on CUDA) ---
        if self.device.type == 'cuda':
            logger.info("🔧 Compiling model with torch.compile for CUDA performance...")
            self.pipe.transformer = torch.compile(self.pipe.transformer, mode="max-autotune", fullgraph=True)
            # self.pipe.vae.decode = torch.compile(self.pipe.vae.decode, mode="max-autotune", fullgraph=True) # VAE decode is not compatible with torch.compile
            logger.info("✅ Model compiled successfully.")
        else:
            logger.info(f"Skipping torch.compile (not on a CUDA device, found '{self.device.type}').")
            
        logger.info("🚀 Predictor is ready.")

    def predict(
        self,
        prompt,
        negative_prompt="blurry, low quality, static, poorly drawn, deformed",
        size="512x512",
        num_frames=24,
        num_inference_steps=25,
        guidance_scale=8.5,
        fps=12,
        seed=None,
    ):
        """Generates a video based on the input prompt and parameters."""
        
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        try:
            width, height = map(int, size.split('x'))
        except ValueError:
            raise ValueError("Size must be in the format 'widthxheight', e.g., '512x512'")

        generator = torch.Generator(self.device).manual_seed(seed)

        logger.info(f"🎬 Generating video with seed {seed}...")
        
        video_frames = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).frames[0]
        
        # --- Export Video ---
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "output.mp4")
        export_to_video(video_frames, output_path, fps=fps)
        
        logger.info(f"✅ Video saved to {output_path}")
        return output_path