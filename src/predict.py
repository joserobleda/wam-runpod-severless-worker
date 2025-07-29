"""
Wan 2.2 TI2V-5B Video Generation Predictor
Optimized for RunPod Serverless deployment with reduced memory footprint
"""

import os
import sys
import gc
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Optional
import torch

# Memory optimization
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Check CUDA availability but don't force it
use_cuda = torch.cuda.is_available()
print(f"CUDA available: {use_cuda}")
if use_cuda:
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Add Wan 2.2 code to Python path
WORKER_DIR = os.environ.get('WORKER_DIR', '/app')
WAN22_CODE_DIR = os.path.join(WORKER_DIR, 'wan2.2_code')
sys.path.insert(0, WAN22_CODE_DIR)

try:
    from PIL import Image
    print("✓ PIL imported successfully")
except ImportError as e:
    print(f"⚠️ PIL import failed: {e}")


class VideoGenerator:
    """Optimized Wan 2.2 TI2V-5B video generator"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.wan22_generate_script = os.path.join(WAN22_CODE_DIR, 'generate.py')
        print(f"Using device: {self.device}")
        print(f"Model path: {self.model_path}")
        print(f"Wan 2.2 generate script: {self.wan22_generate_script}")
        
        # Memory management settings
        if self.device == "cuda":
            # Conservative memory settings for stability
            torch.cuda.empty_cache()
            print(f"GPU memory before setup: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    def setup(self):
        """Setup the generator and verify dependencies"""
        print("Setting up Wan 2.2 TI2V-5B...")
        
        # Check for generate script
        if not os.path.exists(self.wan22_generate_script):
            raise ValueError(f"Wan 2.2 generate script not found: {self.wan22_generate_script}")
        
        # Check model directory
        if not os.path.exists(self.model_path):
            raise ValueError(f"Model path does not exist: {self.model_path}")
        
        # Check for essential model files
        required_files = ['config.json']
        for file in required_files:
            file_path = os.path.join(self.model_path, file)
            if not os.path.exists(file_path):
                print(f"Warning: {file} not found in model directory")
        
        # Test imports
        try:
            import torch
            import transformers
            import diffusers
            print(f"✓ PyTorch {torch.__version__}")
            print(f"✓ Transformers {transformers.__version__}")
            print(f"✓ Diffusers {diffusers.__version__}")
        except ImportError as e:
            print(f"⚠️ Import check failed: {e}")
        
        print("✓ Wan 2.2 TI2V-5B setup complete!")
    
    def preprocess_image(self, image_path: str) -> str:
        """Preprocess input image to ensure compatibility"""
        try:
            if not os.path.exists(image_path):
                raise ValueError(f"Image file not found: {image_path}")
            
            # Load and check image
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
    
    def generate_video(self, prompt: str, image_path: Optional[str] = None,
                      size: str = "1280*704", num_frames: int = 120,
                      guidance_scale: float = 7.0, num_inference_steps: int = 50,
                      seed: Optional[int] = None, fps: int = 24) -> str:
        """Generate video using optimized Wan 2.2 pipeline"""
        
        try:
            print(f"🎬 Generating video with Wan 2.2...")
            print(f"Prompt: {prompt}")
            print(f"Image: {image_path}")
            print(f"Size: {size}, Frames: {num_frames}, Steps: {num_inference_steps}")
            print(f"Guidance: {guidance_scale}, FPS: {fps}, Seed: {seed}")
            
            # Clear GPU memory before generation
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            # Create temporary output directory
            temp_dir = tempfile.mkdtemp(prefix='wan22_output_')
            
            # Build optimized command with memory management flags
            cmd = [
                'python', self.wan22_generate_script,
                '--task', 'ti2v-5B',
                '--size', size,
                '--ckpt_dir', self.model_path,
                '--offload_model', 'True',      # Memory optimization
                '--convert_model_dtype',        # Use efficient dtypes
                '--t5_cpu',                     # Move text encoder to CPU
                '--prompt', prompt,
                '--guidance_scale', str(guidance_scale),
                '--num_inference_steps', str(num_inference_steps),
                '--fps', str(fps)
            ]
            
            # Add optional parameters
            if seed is not None:
                cmd.extend(['--seed', str(seed)])
            
            if image_path:
                processed_image = self.preprocess_image(image_path)
                cmd.extend(['--image', processed_image])
                print(f"Running Image-to-Video generation")
            else:
                print(f"Running Text-to-Video generation")
            
            # Add num_frames if the model supports it
            if hasattr(self, '_supports_num_frames'):
                cmd.extend(['--num_frames', str(num_frames)])
            
            print(f"Executing: {' '.join(cmd)}")
            
            # Change to Wan 2.2 directory and execute
            original_cwd = os.getcwd()
            os.chdir(WAN22_CODE_DIR)
            
            try:
                # Run with timeout and memory monitoring
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=1800,  # 30 minutes max
                    cwd=WAN22_CODE_DIR
                )
                
                if result.returncode != 0:
                    # Enhanced error detection
                    if "CUDA out of memory" in result.stderr or "OutOfMemoryError" in result.stderr:
                        raise RuntimeError(
                            "GPU out of memory. Try reducing video size, frames, or inference steps."
                        )
                    elif "No module named" in result.stderr:
                        raise RuntimeError(f"Missing dependencies: {result.stderr}")
                    elif "FileNotFoundError" in result.stderr:
                        raise RuntimeError(f"File not found: {result.stderr}")
                    else:
                        raise RuntimeError(f"Video generation failed: {result.stderr}")
                
                print(f"✓ Wan 2.2 generation completed successfully")
                print(f"Output: {result.stdout}")
                
            finally:
                os.chdir(original_cwd)
            
            # Enhanced output file detection
            search_dirs = [
                temp_dir,
                WAN22_CODE_DIR,
                os.path.join(WAN22_CODE_DIR, 'outputs'),
                os.path.join(WAN22_CODE_DIR, 'results'),
                os.path.join(WAN22_CODE_DIR, 'generated')
            ]
            
            video_files = []
            for search_dir in search_dirs:
                if os.path.exists(search_dir):
                    for ext in ['.mp4', '.avi', '.mov', '.webm', '.mkv']:
                        # Search current directory
                        video_files.extend(Path(search_dir).glob(f'*{ext}'))
                        # Search subdirectories
                        video_files.extend(Path(search_dir).glob(f'**/*{ext}'))
            
            if not video_files:
                print(f"❌ No video files found!")
                print(f"Searched directories: {search_dirs}")
                for search_dir in search_dirs:
                    if os.path.exists(search_dir):
                        print(f"  {search_dir}: {list(os.listdir(search_dir))}")
                raise RuntimeError("No video file found. Check Wan 2.2 output location.")
            
            # Get the most recent video file
            generated_video = str(max(video_files, key=lambda p: p.stat().st_mtime))
            print(f"✓ Generated video found: {generated_video}")
            
            # Move to final location
            final_output = f"/tmp/video_generation/wan22_output_{os.getpid()}_{seed or 'random'}.mp4"
            os.makedirs(os.path.dirname(final_output), exist_ok=True)
            shutil.move(generated_video, final_output)
            
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            # Final memory cleanup
            if self.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            
            print(f"✓ Video saved to: {final_output}")
            return final_output
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Video generation timed out (30 minutes)")
        except Exception as e:
            print(f"❌ Video generation failed: {e}")
            
            # Emergency memory cleanup
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()
            except:
                pass
                
            raise RuntimeError(f"Video generation failed: {e}")


class Predictor:
    """Main predictor class for RunPod integration"""
    
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.generator = None
    
    def setup(self):
        """Initialize the predictor"""
        print("🚀 Initializing Wan 2.2 TI2V-5B predictor...")
        model_path = os.path.join(self.model_dir, "wan2.2-ti2v-5b")
        self.generator = VideoGenerator(model_path)
        self.generator.setup()
        print("✓ Predictor setup complete")
    
    def predict(self, prompt: str, image_path: Optional[str] = None,
                size: str = "1280*704", num_frames: int = 120,
                guidance_scale: float = 7.0, num_inference_steps: int = 50,
                seed: Optional[int] = None, fps: int = 24,
                use_prompt_extend: bool = False) -> str:
        """Make a prediction"""
        
        try:
            print(f"🎯 Starting prediction...")
            
            # Extend prompt if requested
            if use_prompt_extend:
                prompt = self._extend_prompt(prompt)
                print(f"Extended prompt: '{prompt[:100]}...'")
            
            # Convert size format from 576x1024 to 576*1024 (Wan 2.2 format)
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
            print(f"❌ Prediction failed: {e}")
            
            # Emergency cleanup
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()
            except:
                pass
                
            raise RuntimeError(f"Prediction failed: {e}")
    
    def _extend_prompt(self, prompt: str) -> str:
        """Extend short prompts for better video quality"""
        if len(prompt.strip()) < 50:
            enhanced = f"{prompt.strip()}, cinematic quality, high detail, professional lighting, smooth motion, 720P resolution"
            return enhanced
        return prompt


# Global model instance for reuse
MODEL = None

def setup():
    """Setup function called by RunPod"""
    global MODEL
    if MODEL is None:
        model_dir = os.environ.get('WORKER_MODEL_DIR', '/app/model')
        MODEL = Predictor(model_dir)
        MODEL.setup()
    return MODEL

def predict(prompt: str, **kwargs) -> str:
    """Predict function called by RunPod"""
    global MODEL
    if MODEL is None:
        MODEL = setup()
    return MODEL.predict(prompt, **kwargs) 