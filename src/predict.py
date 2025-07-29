"""
Wan 2.2 TI2V-5B Video Generation for RunPod Serverless
Uses the official Wan 2.2 inference code via subprocess
"""

import os
import sys
import subprocess
import tempfile
import shutil
import torch
import gc
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image

# Add Wan 2.2 code to Python path
WORKER_DIR = os.environ.get('WORKER_DIR', '/app')
WAN22_CODE_DIR = os.path.join(WORKER_DIR, 'wan2.2_code')
sys.path.insert(0, WAN22_CODE_DIR)

use_cuda = os.environ.get('WORKER_USE_CUDA', 'True').lower() == 'true'

class VideoGenerator:
    """
    Wan 2.2 TI2V-5B Video Generation Pipeline
    Supports both Text-to-Video and Image-to-Video generation
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.wan22_generate_script = os.path.join(WAN22_CODE_DIR, 'generate.py')
        print(f"Using device: {self.device}")
        print(f"Model path: {self.model_path}")
        print(f"Wan 2.2 generate script: {self.wan22_generate_script}")
    
    def setup(self):
        """Verify Wan 2.2 setup"""
        try:
            print("Setting up Wan 2.2 TI2V-5B...")
            
            # Check if Wan 2.2 code exists
            if not os.path.exists(self.wan22_generate_script):
                raise ValueError(f"Wan 2.2 generate script not found: {self.wan22_generate_script}")
            
            # Check if model exists
            if not os.path.exists(self.model_path):
                raise ValueError(f"Model path does not exist: {self.model_path}")
            
            # Verify model has required files
            required_files = ['config.json']  # Basic check
            for file in required_files:
                file_path = os.path.join(self.model_path, file)
                if not os.path.exists(file_path):
                    print(f"Warning: {file} not found in model directory")
            
            print("✓ Wan 2.2 TI2V-5B setup complete!")
            
        except Exception as e:
            print(f"Setup failed: {e}")
            raise RuntimeError(f"Wan 2.2 setup failed: {e}")
    
    def preprocess_image(self, image_path: str) -> str:
        """Preprocess input image for I2V generation"""
        try:
            # Verify image exists and is valid
            if not os.path.exists(image_path):
                raise ValueError(f"Image file not found: {image_path}")
            
            # Load and verify image
            img = Image.open(image_path)
            print(f"Input image: {img.size} ({img.mode})")
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
                temp_path = image_path.replace(os.path.splitext(image_path)[1], '_rgb.jpg')
                img.save(temp_path, 'JPEG', quality=95)
                return temp_path
            
            return image_path
            
        except Exception as e:
            print(f"Image preprocessing failed: {e}")
            raise RuntimeError(f"Image preprocessing failed: {e}")
    
    def generate_video(self, 
                      prompt: str,
                      image_path: Optional[str] = None,
                      size: str = "1280*704",  # 720P format for Wan 2.2
                      num_frames: int = 120,
                      guidance_scale: float = 7.0,
                      num_inference_steps: int = 50,
                      seed: Optional[int] = None,
                      fps: int = 24) -> str:
        """Generate video using Wan 2.2 TI2V-5B"""
        
        try:
            print(f"Generating video with Wan 2.2...")
            print(f"Prompt: {prompt}")
            print(f"Image: {image_path}")
            print(f"Size: {size}, Frames: {num_frames}, Steps: {num_inference_steps}")
            print(f"Guidance: {guidance_scale}, FPS: {fps}, Seed: {seed}")
            
            # Create temporary output directory
            temp_dir = tempfile.mkdtemp(prefix='wan22_output_')
            output_path = os.path.join(temp_dir, 'generated_video.mp4')
            
            # Build command for Wan 2.2 generate.py
            cmd = [
                'python', self.wan22_generate_script,
                '--task', 'ti2v-5B',
                '--size', size,
                '--ckpt_dir', self.model_path,
                '--offload_model', 'True',  # For consumer GPUs
                '--convert_model_dtype',    # Memory optimization
                '--t5_cpu',                 # Move T5 to CPU
                '--prompt', prompt,
                '--guidance_scale', str(guidance_scale),
                '--num_inference_steps', str(num_inference_steps),
                '--fps', str(fps)
            ]
            
            # Add optional parameters
            if seed is not None:
                cmd.extend(['--seed', str(seed)])
            
            # Add image for I2V generation
            if image_path:
                processed_image = self.preprocess_image(image_path)
                cmd.extend(['--image', processed_image])
                print(f"Running Image-to-Video generation")
            else:
                print(f"Running Text-to-Video generation")
            
            # Add num_frames parameter (if supported)
            # Note: Check Wan 2.2 docs for exact parameter name
            if hasattr(self, '_supports_num_frames'):
                cmd.extend(['--num_frames', str(num_frames)])
            
            print(f"Running command: {' '.join(cmd)}")
            
            # Set working directory to Wan 2.2 code
            original_cwd = os.getcwd()
            os.chdir(WAN22_CODE_DIR)
            
            try:
                # Run Wan 2.2 generation
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=1800,  # 30 minutes timeout
                    cwd=WAN22_CODE_DIR
                )
                
                if result.returncode != 0:
                    print(f"Wan 2.2 generation failed with return code {result.returncode}")
                    print(f"STDOUT: {result.stdout}")
                    print(f"STDERR: {result.stderr}")
                    # Check for common error patterns
                    if "CUDA out of memory" in result.stderr:
                        raise RuntimeError("GPU out of memory. Try using smaller video size or fewer frames.")
                    elif "No module named" in result.stderr:
                        raise RuntimeError(f"Missing dependencies: {result.stderr}")
                    else:
                        raise RuntimeError(f"Video generation failed: {result.stderr}")
                
                print(f"Wan 2.2 generation completed successfully")
                print(f"Generation output: {result.stdout}")
                
            finally:
                os.chdir(original_cwd)
            
            # Find generated video file in both temp_dir and WAN22_CODE_DIR
            # Wan 2.2 might output to different locations
            search_dirs = [temp_dir, WAN22_CODE_DIR, os.path.join(WAN22_CODE_DIR, 'outputs')]
            video_files = []
            
            for search_dir in search_dirs:
                if os.path.exists(search_dir):
                    for ext in ['.mp4', '.avi', '.mov', '.webm']:
                        video_files.extend(Path(search_dir).glob(f'*{ext}'))
                        video_files.extend(Path(search_dir).glob(f'**/*{ext}'))  # Search subdirectories
            
            if not video_files:
                print(f"Searched directories: {search_dirs}")
                raise RuntimeError(f"No video file found. Check Wan 2.2 output location.")
            
            # Get the most recent video file (in case multiple exist)
            generated_video = str(max(video_files, key=lambda p: p.stat().st_mtime))
            print(f"Generated video: {generated_video}")
            
            # Move to permanent location
            final_output = f"/tmp/video_generation/wan22_output_{os.getpid()}_{seed or 'random'}.mp4"
            os.makedirs(os.path.dirname(final_output), exist_ok=True)
            shutil.move(generated_video, final_output)
            
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            return final_output
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Video generation timed out (30 minutes)")
        except Exception as e:
            print(f"Video generation failed: {e}")
            raise RuntimeError(f"Video generation failed: {e}")

class Predictor:
    """Predictor class for RunPod integration"""
    
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.generator = None
    
    def setup(self):
        """Setup the Wan 2.2 generator"""
        try:
            print("Initializing Wan 2.2 TI2V-5B predictor...")
            
            # Model path
            model_path = os.path.join(self.model_dir, "wan2.2-ti2v-5b")
            
            # Initialize generator
            self.generator = VideoGenerator(model_path)
            self.generator.setup()
            
            print("✓ Predictor setup complete")
            
        except Exception as e:
            print(f"Predictor setup failed: {e}")
            raise
    
    def predict(self,
               prompt: str,
               image_path: Optional[str] = None,
               size: str = "1280*704",
               num_frames: int = 120,
               guidance_scale: float = 7.0,
               num_inference_steps: int = 50,
               seed: Optional[int] = None,
               fps: int = 24,
               use_prompt_extend: bool = False) -> str:
        """Run video generation prediction"""
        
        try:
            print(f"Starting prediction...")
            
            # Extend prompt if requested
            if use_prompt_extend:
                prompt = self._extend_prompt(prompt)
                print(f"Extended prompt: '{prompt[:100]}...'")
            
            # Convert size format from 576x1024 to 1024*576 (Wan 2.2 format)
            if 'x' in size:
                width, height = size.split('x')
                size = f"{width}*{height}"
            
            # Map common sizes to Wan 2.2 supported formats
            size_mapping = {
                "576*1024": "704*1280",    # 9:16 -> 720P vertical
                "1024*576": "1280*704",    # 16:9 -> 720P horizontal
                "512*512": "704*704",      # Square -> closest square
                "768*768": "704*704",      # Square -> closest square
            }
            size = size_mapping.get(size, size)
            
            print(f"Using size: {size}")
            
            # Generate video
            video_path = self.generator.generate_video(
                prompt=prompt,
                image_path=image_path,
                size=size,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed,
                fps=fps
            )
            
            print(f"✓ Video generated successfully: {video_path}")
            return video_path
            
        except Exception as e:
            print(f"Prediction failed: {e}")
            
            # Memory cleanup on error
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()
            except:
                pass
            
            raise RuntimeError(f"Prediction failed: {e}")
    
    def _extend_prompt(self, prompt: str) -> str:
        """Extend/enhance the input prompt for better results"""
        if len(prompt.strip()) < 50:
            # Add some cinematic qualities for short prompts
            enhanced = f"{prompt.strip()}, cinematic quality, high detail, professional lighting, smooth motion, 720P resolution"
            return enhanced
        return prompt 