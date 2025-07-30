"""
Simplified Wan 2.2 TI2V-5B using Diffusers with Full Parameter Support
Based on official Hugging Face documentation
"""

import os
import torch
import numpy as np
from PIL import Image
from diffusers import WanPipeline
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from diffusers.utils import export_to_video
from loguru import logger
import tempfile
import random

# Set matmul precision for better performance on Ampere GPUs
torch.set_float32_matmul_precision('high')

class Predictor:
    """
    A simplified and robust predictor for the Wan2.2-TI2V-5B-Diffusers model.
    """
    def setup(self):
        """Initializes the model pipeline and applies necessary patches."""
        logger.info("🚀 Initializing Wan 2.2 Video Generator...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

        # --- Step 1: Load VAE with mismatch fix ---
        # The model's VAE has a known weights vs. config mismatch.
        # We load it separately with `ignore_mismatched_sizes=True` as the error log suggests.
        logger.info(f"📦 Loading VAE from {model_id} with mismatch fix...")
        vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=self.dtype,
            low_cpu_mem_usage=False,  # Required for this fix
            ignore_mismatched_sizes=True,
            trust_remote_code=True,
        )
        logger.info("✅ VAE loaded successfully.")

        # --- Step 2: Load the Main Pipeline with the Corrected VAE ---
        logger.info(f"📦 Loading WanPipeline from {model_id} with corrected VAE...")
        self.pipe = WanPipeline.from_pretrained(
            model_id,
            vae=vae,
            torch_dtype=self.dtype,
            trust_remote_code=True
        )
        self.pipe.to(self.device)
        logger.info("✅ Pipeline loaded to device.")

        # --- Step 3: Compile for Performance ---
        logger.info("🔧 Compiling model with torch.compile (first run will be slow)...")
        self.pipe.transformer = torch.compile(self.pipe.transformer, mode="max-autotune", fullgraph=True)
        # self.pipe.vae.decode = torch.compile(self.pipe.vae.decode, mode="max-autotune", fullgraph=True) # VAE decode is not compatible with torch.compile
        logger.info("✅ Model compiled successfully.")
        
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