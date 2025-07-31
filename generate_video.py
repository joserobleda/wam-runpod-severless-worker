#!/usr/bin/env python3
"""
Standalone video generation script for local testing.
This matches the structure from the reference repository.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from predict import Predictor
from utils import get_file_size_mb, format_duration
import time

def generate_video(
    prompt: str,
    mode: str = "txt2video",
    negative_prompt: str = "",
    num_frames: int = 48,
    guidance_scale: float = 6.0,
    aspect_ratio: str = "16:9",
    num_inference_steps: int = 50,
    max_sequence_length: int = 226,
    fps: int = 8,
    seed: int = None,
    output_path: str = "generated_video.mp4"
):
    """
    Generate a video from text prompt using CogVideoX-5b.
    
    Args:
        prompt: Text description of the video to generate
        mode: Generation mode (currently only supports 'txt2video')
        negative_prompt: What to avoid in the generation
        num_frames: Number of frames to generate
        guidance_scale: How closely to follow the prompt
        aspect_ratio: Video aspect ratio (informational, not used by CogVideoX)
        num_inference_steps: Number of denoising steps
        max_sequence_length: Maximum sequence length for text encoding
        fps: Frames per second for output video
        seed: Random seed for reproducible results
        output_path: Path to save the generated video
        
    Returns:
        dict: Generation results with video path and metadata
    """
    
    print("🚀 Initializing CogVideoX-5b model...")
    
    # Initialize the predictor
    predictor = Predictor()
    predictor.setup()
    
    print(f"🎬 Starting video generation:")
    print(f"   📝 Prompt: {prompt}")
    print(f"   🎯 Mode: {mode}")
    print(f"   📊 Frames: {num_frames}, Steps: {num_inference_steps}")
    print(f"   🎨 Guidance: {guidance_scale}, FPS: {fps}")
    print(f"   📐 Aspect ratio: {aspect_ratio}")
    if negative_prompt:
        print(f"   ❌ Negative: {negative_prompt}")
    
    start_time = time.time()
    
    try:
        # Generate the video
        encoded_frames = predictor.predict(
            prompt=prompt,
            number_of_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            fps=fps
        )
        
        generation_time = time.time() - start_time
        
        # The predictor creates "new_out.mp4", so we'll move it to the desired output path
        temp_video = "new_out.mp4"
        if os.path.exists(temp_video):
            if output_path != temp_video:
                os.rename(temp_video, output_path)
            
            file_size = get_file_size_mb(output_path)
            
            result = {
                "status": "success",
                "video_path": output_path,
                "generation_time": generation_time,
                "file_size_mb": file_size,
                "parameters": {
                    "mode": mode,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "num_frames": num_frames,
                    "guidance_scale": guidance_scale,
                    "aspect_ratio": aspect_ratio,
                    "num_inference_steps": num_inference_steps,
                    "max_sequence_length": max_sequence_length,
                    "fps": fps,
                    "seed": seed
                }
            }
            
            print(f"✅ Video generation completed!")
            print(f"   📁 Output: {output_path}")
            print(f"   ⏱️  Time: {format_duration(generation_time)}")
            print(f"   📏 Size: {file_size:.1f} MB")
            
            return result
            
        else:
            raise FileNotFoundError("Generated video file not found")
            
    except Exception as e:
        print(f"❌ Generation failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "generation_time": time.time() - start_time
        }

def main():
    """Command line interface for video generation."""
    parser = argparse.ArgumentParser(description="Generate videos using CogVideoX-5b")
    
    parser.add_argument("--prompt", required=True, help="Text prompt for video generation")
    parser.add_argument("--mode", default="txt2video", help="Generation mode (default: txt2video)")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt")
    parser.add_argument("--num-frames", type=int, default=48, help="Number of frames (default: 48)")
    parser.add_argument("--guidance-scale", type=float, default=6.0, help="Guidance scale (default: 6.0)")
    parser.add_argument("--aspect-ratio", default="16:9", help="Aspect ratio (default: 16:9)")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Inference steps (default: 50)")
    parser.add_argument("--max-sequence-length", type=int, default=226, help="Max sequence length (default: 226)")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second (default: 8)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible results")
    parser.add_argument("--output", default="generated_video.mp4", help="Output video path")
    parser.add_argument("--config", help="Load parameters from JSON config file")
    
    args = parser.parse_args()
    
    # Load from config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
            
        # Override with config values, but keep CLI args if provided
        for key, value in config.get('input', {}).items():
            if key == 'num_frames' and not hasattr(args, 'num_frames_set'):
                args.num_frames = value
            elif key == 'guidance_scale' and not hasattr(args, 'guidance_scale_set'):
                args.guidance_scale = value
            # Add other parameter mappings as needed
    
    # Generate the video
    result = generate_video(
        prompt=args.prompt,
        mode=args.mode,
        negative_prompt=args.negative_prompt,
        num_frames=args.num_frames,
        guidance_scale=args.guidance_scale,
        aspect_ratio=args.aspect_ratio,
        num_inference_steps=args.num_inference_steps,
        max_sequence_length=args.max_sequence_length,
        fps=args.fps,
        seed=args.seed,
        output_path=args.output
    )
    
    # Print results
    if result["status"] == "success":
        print(f"\n🎉 Success! Video saved to: {result['video_path']}")
        sys.exit(0)
    else:
        print(f"\n💥 Failed: {result['error']}")
        sys.exit(1)

if __name__ == "__main__":
    main() 