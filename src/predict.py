import os
import sys
import subprocess
import tempfile
import numpy as np
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from PIL import Image
import imageio
import cv2
from pathlib import Path

# Check if we're in the correct directory for Wan2.2
model_dir = os.environ.get('WORKER_MODEL_DIR', '/app/model')
wan_model_path = os.path.join(model_dir, 'wan2.2-ti2v-5b')

# Add the Wan2.2 directory to Python path
if os.path.exists(wan_model_path):
    sys.path.insert(0, wan_model_path)

try:
    # Import Wan2.2 modules
    from diffusers import DiffusionPipeline
    from transformers import AutoTokenizer, AutoModel
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    print("Will attempt to use alternative imports or download dependencies")

use_cuda = os.environ.get('WORKER_USE_CUDA', 'True').lower() == 'true'

class VideoGenerator:
    """
    Wan 2.2 TI2V-5B Video Generation Pipeline
    Supports both Text-to-Video and Image-to-Video generation
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.pipeline = None
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
    
    def setup(self):
        """Load the Wan 2.2 TI2V-5B model"""
        try:
            print("Loading Wan 2.2 TI2V-5B model...")
            
            # Check if model exists
            if not os.path.exists(self.model_path):
                raise ValueError(f"Model path does not exist: {self.model_path}")
            
            # Load the pipeline with the correct model
            # Using DiffusionPipeline for automatic pipeline detection
            self.pipeline = DiffusionPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True,
                variant="fp16" if self.device == "cuda" else None
            )
            
            # Move to appropriate device
            if self.device == "cuda":
                self.pipeline = self.pipeline.to("cuda")
                
                # Enable memory efficient attention if available
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    print("✓ Enabled xformers memory efficient attention")
                except:
                    print("⚠ xformers not available, using default attention")
                
                # Enable CPU offloading for memory efficiency
                try:
                    self.pipeline.enable_model_cpu_offload()
                    print("✓ Enabled model CPU offloading")
                except:
                    print("⚠ Model CPU offloading not available")
            
            print("✓ Wan 2.2 TI2V-5B model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading Wan 2.2 model: {e}")
            # Fallback to manual setup if automatic loading fails
            self._setup_manual()
    
    def _setup_manual(self):
        """Manual setup if automatic pipeline loading fails"""
        try:
            print("Attempting manual model setup...")
            
            # Check if we have the generate.py script from the Wan2.2 repository
            generate_script = os.path.join(self.model_path, "generate.py")
            if os.path.exists(generate_script):
                print("Found generate.py script, using it for inference")
                self.use_script = True
                return
            
            # If no script, try to use the model files directly
            print("No generate.py script found, attempting direct model loading...")
            
            # This is a placeholder for manual model loading
            # In a real implementation, we'd need to understand the exact model architecture
            # For now, we'll use the script-based approach
            self.use_script = False
            
        except Exception as e:
            print(f"Manual setup also failed: {e}")
            raise RuntimeError("Could not set up Wan 2.2 model")
    
    def parse_size(self, size_str: str) -> Tuple[int, int]:
        """Parse size string like '1280*704' into (width, height)"""
        try:
            width, height = map(int, size_str.split('*'))
            return width, height
        except:
            # Default to 720P if parsing fails
            return 1280, 704
    
    def preprocess_image(self, image_path: str, target_size: Tuple[int, int]) -> Image.Image:
        """Preprocess input image for image-to-video generation"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Resize image to target size while maintaining aspect ratio
            width, height = target_size
            image = image.resize((width, height), Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            raise
    
    def generate_video_script(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        size: str = "1280*704",
        num_frames: int = 120,
        guidance_scale: float = 7.0,
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
        fps: int = 24
    ) -> str:
        """Generate video using the Wan2.2 generate.py script"""
        
        try:
            width, height = self.parse_size(size)
            
            # Create temporary output directory
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = os.path.join(temp_dir, "output.mp4")
                
                # Build command
                cmd = [
                    "python", os.path.join(self.model_path, "generate.py"),
                    "--task", "ti2v-5B",
                    "--size", size,
                    "--ckpt_dir", self.model_path,
                    "--prompt", prompt,
                    "--output", output_path
                ]
                
                # Add GPU optimization flags for single GPU
                if self.device == "cuda":
                    cmd.extend([
                        "--offload_model", "True",
                        "--convert_model_dtype",
                        "--t5_cpu"
                    ])
                
                # Add image if provided (for image-to-video)
                if image_path:
                    cmd.extend(["--image", image_path])
                
                # Add seed if provided
                if seed is not None:
                    cmd.extend(["--seed", str(seed)])
                
                print(f"Running command: {' '.join(cmd)}")
                
                # Run the generation
                result = subprocess.run(
                    cmd,
                    cwd=self.model_path,
                    capture_output=True,
                    text=True,
                    timeout=1800  # 30 minute timeout
                )
                
                if result.returncode != 0:
                    print(f"Generation failed with return code {result.returncode}")
                    print(f"STDOUT: {result.stdout}")
                    print(f"STDERR: {result.stderr}")
                    raise RuntimeError(f"Video generation failed: {result.stderr}")
                
                # Check if output file was created
                if not os.path.exists(output_path):
                    raise RuntimeError("Output video file was not created")
                
                print(f"✓ Video generated successfully: {output_path}")
                return output_path
                
        except subprocess.TimeoutExpired:
            raise RuntimeError("Video generation timed out after 30 minutes")
        except Exception as e:
            print(f"Error in script-based generation: {e}")
            raise
    
    def generate_video_pipeline(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        size: str = "1280*704",
        num_frames: int = 120,
        guidance_scale: float = 7.0,
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
        fps: int = 24
    ) -> str:
        """Generate video using the loaded pipeline"""
        
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded")
        
        try:
            width, height = self.parse_size(size)
            
            # Set random seed if provided
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
            
            # Prepare inputs
            generator = torch.Generator(device=self.device)
            if seed is not None:
                generator.manual_seed(seed)
            
            # Generate video
            if image_path:
                # Image-to-video generation
                input_image = self.preprocess_image(image_path, (width, height))
                
                video_frames = self.pipeline(
                    prompt=prompt,
                    image=input_image,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator
                ).frames[0]
            else:
                # Text-to-video generation
                video_frames = self.pipeline(
                    prompt=prompt,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator
                ).frames[0]
            
            # Save video to temporary file
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
                output_path = tmp_file.name
            
            # Convert frames to video
            self._save_video(video_frames, output_path, fps)
            
            print(f"✓ Video generated successfully: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error in pipeline-based generation: {e}")
            raise
    
    def _save_video(self, frames, output_path: str, fps: int = 24):
        """Save video frames to MP4 file"""
        try:
            # Convert frames to numpy arrays if they're PIL Images
            if hasattr(frames[0], 'save'):  # PIL Image
                frames = [np.array(frame) for frame in frames]
            
            # Write video using imageio
            with imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8) as writer:
                for frame in frames:
                    writer.append_data(frame)
                    
        except Exception as e:
            print(f"Error saving video: {e}")
            # Fallback to OpenCV
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))
                
                for frame in frames:
                    # Convert RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                
                out.release()
            except Exception as e2:
                print(f"Both imageio and OpenCV failed: {e2}")
                raise
    
    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        size: str = "1280*704",
        num_frames: int = 120,
        guidance_scale: float = 7.0,
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
        fps: int = 24,
        use_prompt_extend: bool = False
    ) -> str:
        """
        Main generation method that chooses the best available approach
        
        Returns:
            str: Path to the generated video file
        """
        
        print(f"Generating video with prompt: '{prompt[:100]}...'")
        print(f"Parameters: size={size}, frames={num_frames}, steps={num_inference_steps}")
        
        # Extend prompt if requested
        if use_prompt_extend and prompt:
            prompt = self._extend_prompt(prompt)
            print(f"Extended prompt: '{prompt[:100]}...'")
        
        try:
            # Try script-based generation first if available
            if hasattr(self, 'use_script') and self.use_script:
                return self.generate_video_script(
                    prompt=prompt,
                    image_path=image_path,
                    size=size,
                    num_frames=num_frames,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    seed=seed,
                    fps=fps
                )
            
            # Otherwise use pipeline-based generation
            elif self.pipeline is not None:
                return self.generate_video_pipeline(
                    prompt=prompt,
                    image_path=image_path,
                    size=size,
                    num_frames=num_frames,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    seed=seed,
                    fps=fps
                )
            
            else:
                raise RuntimeError("No generation method available")
                
        except Exception as e:
            print(f"Generation failed: {e}")
            raise
    
    def _extend_prompt(self, prompt: str) -> str:
        """Extend/enhance the input prompt for better results"""
        # Basic prompt enhancement - in production you might use a more sophisticated method
        if len(prompt.strip()) < 50:
            # Add some cinematic qualities for short prompts
            enhanced = f"{prompt.strip()}, cinematic quality, high detail, professional lighting, smooth motion"
            return enhanced
        return prompt

class Predictor:
    """Predictor class for RunPod integration"""
    
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.generator = None
    
    def setup(self):
        """Initialize the video generator"""
        try:
            wan_model_path = os.path.join(self.model_dir, 'wan2.2-ti2v-5b')
            self.generator = VideoGenerator(wan_model_path)
            self.generator.setup()
            print("✓ Predictor setup complete")
        except Exception as e:
            print(f"Error in predictor setup: {e}")
            raise
    
    def predict(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        size: str = "1280*704",
        num_frames: int = 120,
        guidance_scale: float = 7.0,
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
        fps: int = 24,
        use_prompt_extend: bool = False
    ) -> str:
        """
        Generate video and return path to output file
        
        Returns:
            str: Path to generated video file
        """
        
        if self.generator is None:
            raise RuntimeError("Generator not initialized. Call setup() first.")
        
        try:
            # Generate video
            video_path = self.generator.generate(
                prompt=prompt,
                image_path=image_path,
                size=size,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed,
                fps=fps,
                use_prompt_extend=use_prompt_extend
            )
            
            return video_path
            
        except Exception as e:
            print(f"Prediction failed: {e}")
            raise
        finally:
            # Memory cleanup
            self._cleanup_memory()
    
    def _cleanup_memory(self):
        """Clean up GPU memory after generation"""
        try:
            if use_cuda and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Get memory stats for monitoring
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_cached = torch.cuda.memory_reserved() / 1024**3      # GB
                print(f"Post-generation GPU Memory - Allocated: {memory_allocated:.2f}GB, Cached: {memory_cached:.2f}GB")
            
            # Force garbage collection
            import gc
            gc.collect()
            
        except Exception as cleanup_error:
            print(f"Warning: Memory cleanup failed: {cleanup_error}") 