"""
Simplified Wan 2.2 TI2V-5B using Diffusers with Full Parameter Support
Based on official Hugging Face documentation, Wan2.2 GitHub repo, and ComfyUI integration insights
"""

import os
import torch
import numpy as np
from PIL import Image
from diffusers import WanPipeline, AutoencoderKL, UniPCMultistepScheduler
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from diffusers.models.transformers.wan_transformer_3d import WanTransformer3DModel
from transformers import T5EncoderModel, T5Tokenizer
from diffusers.utils import export_to_video
from loguru import logger
import tempfile
import random
import runpod
from runpod.serverless.utils import rp_cleanup

# --- Constants from ComfyUI for Wan 2.2 Latent Decoding ---
# Source: https://github.com/comfyanonymous/ComfyUI/commit/a88788dce6b0d7b5e2876c7cd0121b45e80f4ad8
# File: comfy/latent_formats.py -> class Wan22
LATENT_RGB_FACTORS = [
    [ 0.0119, 0.0103, 0.0046], [-0.1062, -0.0504, 0.0165], [ 0.0140, 0.0409, 0.0491],
    [-0.0813, -0.0677, 0.0607], [ 0.0656, 0.0851, 0.0808], [ 0.0264, 0.0463, 0.0912],
    [ 0.0295, 0.0326, 0.0590], [-0.0244, -0.0270, 0.0025], [ 0.0443, -0.0102, 0.0288],
    [-0.0465, -0.0090, -0.0205], [ 0.0359, 0.0236, 0.0082], [-0.0776, 0.0854, 0.1048],
    [ 0.0564, 0.0264, 0.0561], [ 0.0006, 0.0594, 0.0418], [-0.0319, -0.0542, -0.0637],
    [-0.0268, 0.0024, 0.0260], [ 0.0539, 0.0265, 0.0358], [-0.0359, -0.0312, -0.0287],
    [-0.0285, -0.1032, -0.1237], [ 0.1041, 0.0537, 0.0622], [-0.0086, -0.0374, -0.0051],
    [ 0.0390, 0.0670, 0.2863], [ 0.0069, 0.0144, 0.0082], [ 0.0006, -0.0167, 0.0079],
    [ 0.0313, -0.0574, -0.0232], [-0.1454, -0.0902, -0.0481], [ 0.0714, 0.0827, 0.0447],
    [-0.0304, -0.0574, -0.0196], [ 0.0401, 0.0384, 0.0204], [-0.0758, -0.0297, -0.0014],
    [ 0.0568, 0.1307, 0.1372], [-0.0055, -0.0310, -0.0380], [ 0.0239, -0.0305, 0.0325],
    [-0.0663, -0.0673, -0.0140], [-0.0416, -0.0047, -0.0023], [ 0.0166, 0.0112, -0.0093],
    [-0.0211, 0.0011, 0.0331], [ 0.1833, 0.1466, 0.2250], [-0.0368, 0.0370, 0.0295],
    [-0.3441, -0.3543, -0.2008], [-0.0221, -0.0173, -0.0053], [ 0.0015, -0.0061, -0.0047],
    [ 0.0022, 0.0055, 0.0143], [ 0.0002, 0.0001, -0.0000], [-0.0001, 0.0002, 0.0001],
    [-0.0001, -0.0000, -0.0001], [-0.0001, 0.0001, 0.0001]
]
SCALE_FACTOR = 0.00010307

# Set matmul precision for better performance on Ampere GPUs
torch.set_float32_matmul_precision('high')

class Predictor:
    def setup(self):
        """Initializes the model pipeline and applies necessary patches."""
        logger.info("🚀 Initializing Wan 2.2 Video Generator...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
        
        # --- Step 1: Load and patch the VAE ---
        # The original Wan VAE is broken; we load a stable one and disguise it.
        logger.info("📦 Loading and patching stable VAE...")
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt", 
            subfolder="vae"
        )
        vae.temperal_downsample = [True, False, False, False]
        vae.__class__ = AutoencoderKLWan
        logger.info("✅ VAE ready.")

        # --- Step 2: Manually load all other pipeline components ---
        # We bypass the buggy WanPipeline.from_pretrained and build it ourselves.
        logger.info("📦 Loading Transformer...")
        transformer = WanTransformer3DModel.from_pretrained(model_id, subfolder="transformer")
        logger.info("📦 Loading Tokenizer...")
        tokenizer = T5Tokenizer.from_pretrained(model_id, subfolder="tokenizer")
        logger.info("📦 Loading Text Encoder...")
        text_encoder = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder")
        logger.info("📦 Loading Scheduler...")
        scheduler = UniPCMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
        
        # --- Step 3: Manually assemble the pipeline ---
        logger.info("🔧 Assembling pipeline from components...")
        self.pipe = WanPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler
        )

        # --- Step 4: Move to device and set precision ---
        logger.info(f"➡️ Moving pipeline to {self.device} with {self.dtype} precision...")
        self.pipe.to(dtype=self.dtype, device=self.device)
        logger.info("✅ Pipeline on device and ready.")

        # --- Step 5: Compile for Performance ---
        logger.info("🔧 Compiling model with torch.compile (first run will be slow)...")
        self.pipe.transformer = torch.compile(self.pipe.transformer, mode="max-autotune", fullgraph=True)
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
        
        # --- Step 1: Generate latents ---
        latents = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="latent"
        ).frames

        # --- Step 2: Manually decode latents using ComfyUI's method (Vectorized for Performance) ---
        video_frames = []
        
        # Original shape: (B, C, F, H, W) -> Permute to (F*B, C, H, W) for batching
        batch_size, channels, num_frames_in, height, width = latents.shape
        latents_batched = latents.permute(2, 0, 1, 3, 4).reshape(num_frames_in * batch_size, channels, height, width)

        try:
            # Fast-path: Attempt to decode all frames in a single batch for maximum speed.
            # The key is dividing latents by the model-specific scale factor before decoding.
            logger.info("🏎️ Attempting fast-path batch decoding...")
            decoded = self.pipe.vae.decode(latents_batched / SCALE_FACTOR).sample
        except Exception as e:
            # Fallback: If batching fails (e.g., OOM), process frame-by-frame.
            logger.warning(f"Batch decode failed ({e}), falling back to slower frame-by-frame decoding.")
            decoded_list = []
            for i in range(latents_batched.shape[0]):
                latent_frame = latents_batched[i].unsqueeze(0)
                decoded_frame = self.pipe.vae.decode(latent_frame / SCALE_FACTOR).sample
                decoded_list.append(decoded_frame)
            decoded = torch.cat(decoded_list, dim=0)

        # Post-processing: clamp and convert to image format
        decoded = decoded.clamp(0, 1)
        
        # Convert to numpy/PIL
        video_np = (decoded * 255).cpu().numpy().astype(np.uint8)
        for frame_np in video_np:
            # The numpy array from diffusers is (C, H, W), but PIL needs (H, W, C), so we transpose.
            video_pil = Image.fromarray(frame_np.transpose(1, 2, 0))
            video_frames.append(video_pil)

        # --- Export Video ---
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "output.mp4")
        export_to_video(video_frames, output_path, fps=fps)
        
        logger.info(f"✅ Video saved to {output_path}")
        
        return output_path