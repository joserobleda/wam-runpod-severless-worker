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
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Memory optimization
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Check CUDA availability but don't force it
use_cuda = torch.cuda.is_available()
logger.info(f"CUDA available: {use_cuda}")
if use_cuda:
    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Add Wan 2.2 code to Python path
WORKER_DIR = os.environ.get('WORKER_DIR', '/app')
WAN22_CODE_DIR = os.path.join(WORKER_DIR, 'wan2.2_code')
sys.path.insert(0, WAN22_CODE_DIR)

try:
    from PIL import Image
    logger.info("✓ PIL imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ PIL import failed: {e}")


def get_python_executable():
    """Get the correct Python executable path"""
    # Try to find the correct Python executable
    python_candidates = [
        sys.executable,  # Current Python interpreter - highest priority
        '/usr/bin/python3',
        '/usr/local/bin/python3',
        '/opt/conda/bin/python',
        '/app/venv/bin/python',
        'python3',       # Try in PATH
        'python',        # Fallback
    ]
    
    for candidate in python_candidates:
        try:
            if not candidate:
                continue
                
            # If it's an absolute path, check if file exists
            if os.path.isabs(candidate):
                if os.path.exists(candidate) and os.access(candidate, os.X_OK):
                    logger.info(f"Found Python executable: {candidate}")
                    return candidate
            else:
                # Test if command is available in PATH
                result = subprocess.run(['which', candidate], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and result.stdout.strip():
                    executable_path = result.stdout.strip()
                    logger.info(f"Found Python executable in PATH: {candidate} -> {executable_path}")
                    return candidate
        except Exception as e:
            logger.debug(f"Failed to test candidate {candidate}: {e}")
            continue
    
    # If all else fails, use sys.executable
    logger.warning(f"Using fallback Python executable: {sys.executable}")
    return sys.executable


class VideoGenerator:
    """Optimized Wan 2.2 TI2V-5B video generator"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.wan22_generate_script = os.path.join(WAN22_CODE_DIR, 'generate.py')
        self.python_executable = get_python_executable()
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Wan 2.2 generate script: {self.wan22_generate_script}")
        logger.info(f"Python executable: {self.python_executable}")
        
        # Memory management settings
        if self.device == "cuda":
            # Conservative memory settings for stability
            torch.cuda.empty_cache()
            logger.info(f"GPU memory before setup: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    def setup(self):
        """Setup the generator and verify dependencies"""
        logger.info("Setting up Wan 2.2 TI2V-5B...")
        
        # Check if WAN22_CODE_DIR exists
        if not os.path.exists(WAN22_CODE_DIR):
            raise ValueError(f"Wan 2.2 code directory not found: {WAN22_CODE_DIR}")
        
        # Check for generate script
        if not os.path.exists(self.wan22_generate_script):
            # Try alternative locations
            alternative_scripts = [
                os.path.join(WAN22_CODE_DIR, 'scripts', 'generate.py'),
                os.path.join(WAN22_CODE_DIR, 'src', 'generate.py'),
                os.path.join(WAN22_CODE_DIR, 'wan22', 'generate.py'),
            ]
            
            found_script = None
            for alt_script in alternative_scripts:
                if os.path.exists(alt_script):
                    found_script = alt_script
                    break
            
            if found_script:
                self.wan22_generate_script = found_script
                logger.info(f"Found generate script at alternative location: {found_script}")
            else:
                # List contents of WAN22_CODE_DIR for debugging
                try:
                    contents = os.listdir(WAN22_CODE_DIR)
                    logger.error(f"Contents of {WAN22_CODE_DIR}: {contents}")
                except Exception as e:
                    logger.error(f"Cannot list contents of {WAN22_CODE_DIR}: {e}")
                
                raise ValueError(f"Wan 2.2 generate script not found. Searched: {self.wan22_generate_script} and alternatives: {alternative_scripts}")
        
        # Check model directory
        if not os.path.exists(self.model_path):
            # Try alternative model paths
            alternative_paths = [
                os.path.join(os.path.dirname(self.model_path), 'models', 'wan2.2-ti2v-5b'),
                os.path.join(WORKER_DIR, 'models', 'wan2.2-ti2v-5b'),
                os.path.join('/app', 'models', 'wan2.2-ti2v-5b'),
            ]
            
            found_model = None
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    found_model = alt_path
                    break
            
            if found_model:
                self.model_path = found_model
                logger.info(f"Found model at alternative location: {found_model}")
            else:
                raise ValueError(f"Model path does not exist: {self.model_path}. Also tried: {alternative_paths}")
        
        # Check for essential model files (more flexible)
        essential_indicators = ['config.json', 'model_index.json', 'pytorch_model.bin', 'model.safetensors']
        found_indicators = []
        for indicator in essential_indicators:
            indicator_path = os.path.join(self.model_path, indicator)
            if os.path.exists(indicator_path):
                found_indicators.append(indicator)
        
        if not found_indicators:
            logger.warning(f"No standard model files found in {self.model_path}")
            # List model directory contents
            try:
                model_contents = os.listdir(self.model_path)
                logger.info(f"Model directory contents: {model_contents}")
            except Exception as e:
                logger.error(f"Cannot list model directory: {e}")
        else:
            logger.info(f"Found model indicators: {found_indicators}")
        
        # Test Python executable with more robust checking
        try:
            result = subprocess.run([self.python_executable, '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info(f"✓ Python executable working: {result.stdout.strip()}")
            else:
                logger.warning(f"⚠️ Python executable test failed: {result.stderr}")
                # Try with python3 if the current one fails
                if 'python3' not in self.python_executable:
                    try:
                        result2 = subprocess.run(['python3', '--version'], 
                                               capture_output=True, text=True, timeout=10)
                        if result2.returncode == 0:
                            self.python_executable = 'python3'
                            logger.info(f"✓ Switched to python3: {result2.stdout.strip()}")
                    except:
                        pass
        except Exception as e:
            logger.warning(f"⚠️ Python executable test error: {e}")
        
        # Test imports with more detailed error reporting
        try:
            import torch
            import transformers
            import diffusers
            logger.info(f"✓ PyTorch {torch.__version__}")
            logger.info(f"✓ Transformers {transformers.__version__}")
            logger.info(f"✓ Diffusers {diffusers.__version__}")
        except ImportError as e:
            logger.error(f"⚠️ Import check failed: {e}")
            # This is critical, so we should raise an exception
            raise RuntimeError(f"Required dependencies not available: {e}")
        
        logger.info("✓ Wan 2.2 TI2V-5B setup complete!")
    
    def preprocess_image(self, image_path: str) -> str:
        """Preprocess input image to ensure compatibility"""
        try:
            if not os.path.exists(image_path):
                raise ValueError(f"Image file not found: {image_path}")
            
            # Load and check image
            img = Image.open(image_path)
            logger.info(f"Input image: {img.size} ({img.mode})")
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
                # Create temp file in a more reliable location
                temp_dir = tempfile.gettempdir()
                temp_filename = f"rgb_image_{os.getpid()}_{hash(image_path) % 10000}.jpg"
                temp_path = os.path.join(temp_dir, temp_filename)
                img.save(temp_path, 'JPEG', quality=95)
                logger.info(f"Converted image to RGB: {temp_path}")
                return temp_path
            
            return image_path
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise RuntimeError(f"Image preprocessing failed: {e}")
    
    def generate_video(self, prompt: str, image_path: Optional[str] = None,
                      size: str = "1280*704", num_frames: int = 120,
                      guidance_scale: float = 7.0, num_inference_steps: int = 50,
                      seed: Optional[int] = None, fps: int = 24) -> str:
        """Generate video using optimized Wan 2.2 pipeline"""
        
        try:
            logger.info(f"🎬 Generating video with Wan 2.2...")
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Image: {image_path}")
            logger.info(f"Size: {size}, Frames: {num_frames}, Steps: {num_inference_steps}")
            logger.info(f"Guidance: {guidance_scale}, FPS: {fps}, Seed: {seed}")
            
            # Clear GPU memory before generation
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            # Create temporary output directory in a more reliable location
            temp_dir = tempfile.mkdtemp(prefix='wan22_output_', dir='/tmp')
            logger.info(f"Created temp directory: {temp_dir}")
            
            # Build optimized command with memory management flags
            cmd = [
                self.python_executable,
                self.wan22_generate_script,
                '--task', 'ti2v-5B',
                '--size', size,
                '--ckpt_dir', self.model_path,
                '--offload_model', 'True',
                '--convert_model_dtype',
                '--t5_cpu',
                '--prompt', prompt,
                '--guidance_scale', str(guidance_scale),
                '--num_inference_steps', str(num_inference_steps),
                '--fps', str(fps)
            ]
            
            # Add output directory to command if supported
            cmd.extend(['--output_dir', temp_dir])
            
            # Add optional parameters
            if seed is not None:
                cmd.extend(['--seed', str(seed)])
            
            if image_path:
                processed_image = self.preprocess_image(image_path)
                cmd.extend(['--image', processed_image])
                logger.info(f"Running Image-to-Video generation")
            else:
                logger.info(f"Running Text-to-Video generation")
            
            # Add num_frames parameter (most models support this)
            cmd.extend(['--num_frames', str(num_frames)])
            
            logger.info(f"Executing command: {' '.join(cmd[:10])}... (truncated)")
            
            # Change to Wan 2.2 directory and execute
            original_cwd = os.getcwd()
            
            try:
                os.chdir(WAN22_CODE_DIR)
                logger.info(f"Changed to directory: {WAN22_CODE_DIR}")
                
                # Set environment variables for better compatibility
                env = os.environ.copy()
                env['PYTHONPATH'] = f"{WAN22_CODE_DIR}:{env.get('PYTHONPATH', '')}"
                env['CUDA_VISIBLE_DEVICES'] = env.get('CUDA_VISIBLE_DEVICES', '0')
                
                # Run with timeout and memory monitoring
                logger.info("Starting video generation subprocess...")
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=1800,  # 30 minutes max
                    cwd=WAN22_CODE_DIR,
                    env=env
                )
                
                logger.info(f"Subprocess completed with return code: {result.returncode}")
                
                if result.returncode != 0:
                    logger.error(f"Command failed with stderr: {result.stderr}")
                    logger.error(f"Command stdout: {result.stdout}")
                    
                    # Enhanced error detection
                    error_msg = result.stderr.lower()
                    if "cuda out of memory" in error_msg or "outofmemoryerror" in error_msg:
                        raise RuntimeError(
                            "GPU out of memory. Try reducing video size, frames, or inference steps."
                        )
                    elif "no module named" in error_msg:
                        raise RuntimeError(f"Missing dependencies: {result.stderr}")
                    elif "filenotfounderror" in error_msg or "no such file" in error_msg:
                        raise RuntimeError(f"File not found: {result.stderr}")
                    elif "permission denied" in error_msg:
                        raise RuntimeError(f"Permission denied: {result.stderr}")
                    else:
                        raise RuntimeError(f"Video generation failed: {result.stderr}")
                
                logger.info(f"✓ Wan 2.2 generation completed successfully")
                if result.stdout:
                    logger.info(f"Subprocess output: {result.stdout}")
                
            finally:
                os.chdir(original_cwd)
                logger.info(f"Restored directory: {original_cwd}")
            
            # Enhanced output file detection with more comprehensive search
            search_dirs = [
                temp_dir,
                WAN22_CODE_DIR,
                os.path.join(WAN22_CODE_DIR, 'outputs'),
                os.path.join(WAN22_CODE_DIR, 'results'),
                os.path.join(WAN22_CODE_DIR, 'generated'),
                os.path.join(WAN22_CODE_DIR, 'output'),
                '/tmp',
                os.getcwd()  # Current working directory
            ]
            
            video_files = []
            for search_dir in search_dirs:
                if os.path.exists(search_dir):
                    logger.info(f"Searching for videos in: {search_dir}")
                    for ext in ['.mp4', '.avi', '.mov', '.webm', '.mkv']:
                        # Search current directory
                        pattern_files = list(Path(search_dir).glob(f'*{ext}'))
                        video_files.extend(pattern_files)
                        # Search subdirectories (limit depth to avoid infinite recursion)
                        try:
                            subdir_files = list(Path(search_dir).glob(f'*/*{ext}'))
                            video_files.extend(subdir_files)
                        except:
                            pass  # Skip if there are permission issues
            
            # Remove duplicates and sort by modification time
            video_files = list(set(video_files))
            
            if not video_files:
                logger.error(f"❌ No video files found!")
                logger.error(f"Searched directories: {search_dirs}")
                for search_dir in search_dirs:
                    if os.path.exists(search_dir):
                        try:
                            files = list(os.listdir(search_dir))[:10]  # Limit to first 10 files
                            logger.error(f"  {search_dir}: {files}")
                        except PermissionError:
                            logger.error(f"  {search_dir}: Permission denied")
                        except Exception as e:
                            logger.error(f"  {search_dir}: Error listing - {e}")
                
                raise RuntimeError("No video file found. Check Wan 2.2 output location and permissions.")
            
            # Get the most recent video file
            try:
                generated_video = str(max(video_files, key=lambda p: p.stat().st_mtime))
                logger.info(f"✓ Generated video found: {generated_video}")
                
                # Verify the file is not empty
                file_size = os.path.getsize(generated_video)
                if file_size == 0:
                    raise RuntimeError(f"Generated video file is empty: {generated_video}")
                
                logger.info(f"Video file size: {file_size / (1024*1024):.2f} MB")
                
            except Exception as e:
                logger.error(f"Error accessing video files: {e}")
                # Just take the first one if stat fails
                generated_video = str(video_files[0])
                logger.info(f"Using first available video: {generated_video}")
            
            # Move to final location with better error handling
            final_output_dir = "/tmp/video_generation"
            os.makedirs(final_output_dir, exist_ok=True)
            
            final_output = os.path.join(final_output_dir, f"wan22_output_{os.getpid()}_{seed or 'random'}.mp4")
            
            try:
                shutil.move(generated_video, final_output)
                logger.info(f"✓ Video moved to final location: {final_output}")
            except Exception as e:
                logger.error(f"Failed to move video, copying instead: {e}")
                shutil.copy2(generated_video, final_output)
                logger.info(f"✓ Video copied to final location: {final_output}")
            
            # Cleanup temporary files
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.info("✓ Cleaned up temporary directory")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory: {e}")
            
            # Final memory cleanup
            if self.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            
            logger.info(f"✓ Video generation completed: {final_output}")
            return final_output
            
        except subprocess.TimeoutExpired:
            logger.error("Video generation timed out (30 minutes)")
            raise RuntimeError("Video generation timed out (30 minutes)")
        except Exception as e:
            logger.error(f"❌ Video generation failed: {e}")
            
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
        logger.info("🚀 Initializing Wan 2.2 TI2V-5B predictor...")
        
        # More flexible model path detection
        possible_model_paths = [
            os.path.join(self.model_dir, "wan2.2-ti2v-5b"),
            os.path.join(self.model_dir, "wan2.2_ti2v_5b"),
            os.path.join(self.model_dir, "wan22-ti2v-5b"),
            self.model_dir,  # Model files might be directly in model_dir
        ]
        
        model_path = None
        for path in possible_model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            logger.error(f"No model found in any of these locations: {possible_model_paths}")
            raise ValueError(f"Model not found. Searched: {possible_model_paths}")
        
        logger.info(f"Using model path: {model_path}")
        self.generator = VideoGenerator(model_path)
        self.generator.setup()
        logger.info("✓ Predictor setup complete")
    
    def predict(self, prompt: str, image_path: Optional[str] = None,
                size: str = "1280*704", num_frames: int = 120,
                guidance_scale: float = 7.0, num_inference_steps: int = 50,
                seed: Optional[int] = None, fps: int = 24,
                use_prompt_extend: bool = False) -> str:
        """Make a prediction"""
        
        try:
            logger.info(f"🎯 Starting prediction...")
            logger.info(f"Parameters: size={size}, frames={num_frames}, steps={num_inference_steps}")
            
            # Validate inputs
            if not prompt or not prompt.strip():
                raise ValueError("Prompt cannot be empty")
            
            if image_path and not os.path.exists(image_path):
                raise ValueError(f"Image file not found: {image_path}")
            
            # Extend prompt if requested
            if use_prompt_extend:
                prompt = self._extend_prompt(prompt)
                logger.info(f"Extended prompt: '{prompt[:100]}...'")
            
            # Convert size format from 576x1024 to 576*1024 (Wan 2.2 format)
            if 'x' in size and '*' not in size:
                width, height = size.split('x')
                size = f"{width}*{height}"
            
            # Map common sizes to Wan 2.2 supported formats
            size_mapping = {
                "576*1024": "704*1280",    # 9:16 -> 720P vertical
                "1024*576": "1280*704",    # 16:9 -> 720P horizontal  
                "512*512": "704*704",      # Square -> closest square
                "768*768": "704*704",      # Square -> closest square
                "640*480": "704*576",      # 4:3 aspect ratio
                "480*640": "576*704",      # 3:4 aspect ratio
            }
            size = size_mapping.get(size, size)
            
            logger.info(f"Using size: {size}")
            
            # Validate parameters
            if num_frames <= 0 or num_frames > 300:
                logger.warning(f"Unusual num_frames: {num_frames}, using 120")
                num_frames = 120
                
            if guidance_scale <= 0 or guidance_scale > 20:
                logger.warning(f"Unusual guidance_scale: {guidance_scale}, using 7.0")
                guidance_scale = 7.0
                
            if num_inference_steps <= 0 or num_inference_steps > 100:
                logger.warning(f"Unusual inference_steps: {num_inference_steps}, using 50")
                num_inference_steps = 50
            
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
            
            # Verify output file exists and is not empty
            if not os.path.exists(video_path):
                raise RuntimeError(f"Generated video file not found: {video_path}")
            
            file_size = os.path.getsize(video_path)
            if file_size == 0:
                raise RuntimeError(f"Generated video file is empty: {video_path}")
            
            logger.info(f"✓ Video generated successfully: {video_path} ({file_size / (1024*1024):.2f} MB)")
            return video_path
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            
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
        prompt = prompt.strip()
        if len(prompt) < 50:
            enhanced = f"{prompt}, cinematic quality, high detail, professional lighting, smooth motion, 720P resolution"
            return enhanced
        return prompt


# Global model instance for reuse
MODEL = None

def setup():
    """Setup function called by RunPod"""
    global MODEL
    try:
        if MODEL is None:
            model_dir = os.environ.get('WORKER_MODEL_DIR', '/app/model')
            logger.info(f"Setting up model from directory: {model_dir}")
            MODEL = Predictor(model_dir)
            MODEL.setup()
        return MODEL
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        raise

def predict(prompt: str, **kwargs) -> str:
    """Predict function called by RunPod"""
    global MODEL
    try:
        if MODEL is None:
            logger.info("Model not initialized, setting up...")
            MODEL = setup()
        return MODEL.predict(prompt, **kwargs)
    except Exception as e:
        logger.error(f"Predict function failed: {e}")
        raise
