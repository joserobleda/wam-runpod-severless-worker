import os
from typing import List
from utils import encode_video_to_base64
import torch
from diffusers import CogVideoXPipeline, AutoencoderKLCogVideoX, CogVideoXTransformer3DModel
from diffusers.utils import export_to_video
from transformers import T5EncoderModel, T5Tokenizer

model_path = 'model_cache'   # The local directory to save downloaded checkpoint
model_id = model_path

MODEL_ID = "THUDM/CogVideoX-5b"
MODEL_CACHE = "diffusers-cache"


class Predictor:
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        transformer = CogVideoXTransformer3DModel.from_pretrained(
            model_id, 
            subfolder="transformer",
            torch_dtype=torch.float16
        )
        text_encoder = T5EncoderModel.from_pretrained(
            model_id, 
            subfolder="text_encoder", 
            torch_dtype=torch.float16
        )
        vae = AutoencoderKLCogVideoX.from_pretrained(
            model_id, 
            subfolder="vae", 
            torch_dtype=torch.float16
        )
        tokenizer = T5Tokenizer.from_pretrained(model_id, subfolder="tokenizer")
        
        self.pipe = CogVideoXPipeline.from_pretrained(
            model_id, 
            tokenizer=tokenizer, 
            text_encoder=text_encoder,
            transformer=transformer, 
            vae=vae, 
            torch_dtype=torch.float16
        ).to("cuda")
        
        # Try to enable xformers memory efficient attention if available
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("✅ xFormers memory efficient attention enabled")
        except Exception as e:
            print(f"⚠️ xFormers not available or failed to enable: {e}")
            print("📝 Continuing with standard attention (this is fine)")
        
        print("🎬 CogVideoX pipeline initialized successfully")

    @torch.inference_mode()
    def predict(self, prompt, number_of_frames, num_inference_steps, guidance_scale, fps):
        if torch.cuda.is_available():
            print('=============cuda available==================')
            generator = torch.Generator('cuda').manual_seed(42)
        else:
            print('=============cuda not available==============')
            generator = torch.Generator().manual_seed(42)
        
        print('inference')
        video = self.pipe(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=num_inference_steps,
            num_frames=number_of_frames,
            guidance_scale=guidance_scale,
            generator=generator,
        ).frames[0]

        file_name = "new_out.mp4"
        export_to_video(video, file_name, fps=fps)

        encoded_frames = encode_video_to_base64(file_name)
        return encoded_frames

