import io
import os
import argparse
import base64
import tempfile
import shutil
from pathlib import Path
from typing import Optional
# runpod utils
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils.rp_upload import upload_in_memory_object
from runpod.serverless.utils import rp_download, rp_cleanup
# predictor
import predict
from rp_schema import INPUT_SCHEMA
# direct S3 upload
import boto3
from botocore.exceptions import ClientError


# Model params
model_dir = os.getenv("WORKER_MODEL_DIR", "/model")


def upload_video(video_path: str, key: str):
    """Uploads video to Cloudflare R2 bucket if available, otherwise returns base64 encoded video."""
    
    try:
        # Read video file
        with open(video_path, 'rb') as f:
            video_bytes = f.read()
            
        print(f"Video file size: {len(video_bytes) / 1024 / 1024:.2f} MB")
        
        # Upload to Cloudflare R2 (S3-compatible) - Direct upload
        if os.environ.get('BUCKET_ENDPOINT_URL', False):
            try:
                # Parse endpoint and bucket from URL
                endpoint_url = os.environ.get('BUCKET_ENDPOINT_URL')
                
                # If the URL contains a bucket name, extract it
                if '/' in endpoint_url.split('://', 1)[1]:
                    # URL format: https://account-id.r2.cloudflarestorage.com/bucket-name
                    base_url, bucket_name = endpoint_url.rsplit('/', 1)
                    actual_endpoint = base_url
                else:
                    # URL format: https://account-id.r2.cloudflarestorage.com
                    actual_endpoint = endpoint_url
                    bucket_name = os.environ.get('BUCKET_NAME', 'wam-videos')
                
                # Create S3 client for direct upload
                s3_client = boto3.client(
                    's3',
                    endpoint_url=actual_endpoint,
                    aws_access_key_id=os.environ.get('BUCKET_ACCESS_KEY_ID'),
                    aws_secret_access_key=os.environ.get('BUCKET_SECRET_ACCESS_KEY'),
                    region_name='auto'  # Cloudflare R2 uses 'auto' as region
                )
                
                # Upload directly to root of bucket
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=key,  # File will go directly to root with this key
                    Body=video_bytes,
                    ContentType='video/mp4'
                )
                
                # Return the public URL
                public_url = f"{actual_endpoint}/{bucket_name}/{key}"
                print(f"✓ Video uploaded to R2: {public_url}")
                return public_url
                
            except Exception as e:
                print(f"Direct S3 upload failed: {e}")
                print("Falling back to RunPod upload function...")
                # Fallback to original RunPod function
                return upload_in_memory_object(
                    key,
                    video_bytes,
                    bucket_creds = {
                        "endpointUrl": os.environ.get('BUCKET_ENDPOINT_URL', None),
                        "accessId": os.environ.get('BUCKET_ACCESS_KEY_ID', None),
                        "accessSecret": os.environ.get('BUCKET_SECRET_ACCESS_KEY', None)
                    }
                )
        
        # Base64 encode for direct return (fallback)
        print("⚠ No R2 credentials found, returning base64 encoded video")
        return base64.b64encode(video_bytes).decode('utf-8')
        
    except Exception as e:
        print(f"Error reading/uploading video file: {e}")
        raise

def download_image(image_url: str, job_id: str) -> Optional[str]:
    """Download image from URL and return local path"""
    if not image_url:
        return None
        
    try:
        print(f"Downloading image from: {image_url}")
        downloaded_files = rp_download.download_files_from_urls(job_id, [image_url])
        if downloaded_files and len(downloaded_files) > 0:
            image_path = downloaded_files[0]
            print(f"✓ Image downloaded to: {image_path}")
            return image_path
        else:
            print("⚠ No files downloaded")
            return None
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

def run(job):
    """Main job processing function"""
    try:
        job_input = job['input']
        job_id = job['id']
        print(f"Processing job {job_id} with input: {job_input}")

        # Input validation
        validated_input = validate(job_input, INPUT_SCHEMA)

        if 'errors' in validated_input:
            print(f"Validation errors: {validated_input['errors']}")
            return {"error": validated_input['errors']}
        validated_input = validated_input['validated_input']

        # Extract parameters
        prompt = validated_input['prompt']
        image_url = validated_input.get('image')
        size = validated_input.get('size', '1280*704')
        num_frames = validated_input.get('num_frames', 120)
        guidance_scale = validated_input.get('guidance_scale', 7.0)
        num_inference_steps = validated_input.get('num_inference_steps', 50)
        seed = validated_input.get('seed')
        fps = validated_input.get('fps', 24)
        use_prompt_extend = validated_input.get('use_prompt_extend', False)

        print(f"Generation parameters:")
        print(f"  - Prompt: '{prompt[:100]}...'")
        print(f"  - Image URL: {image_url}")
        print(f"  - Size: {size}")
        print(f"  - Frames: {num_frames}")
        print(f"  - Steps: {num_inference_steps}")
        print(f"  - Guidance: {guidance_scale}")
        print(f"  - Seed: {seed}")
        print(f"  - FPS: {fps}")

        # Download input image if provided
        image_path = None
        if image_url:
            image_path = download_image(image_url, job_id)
            if not image_path:
                return {"error": "Failed to download input image"}

        # Generate video
        print("Starting video generation...")
        try:
            video_path = MODEL.predict(
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
            
            if not video_path or not os.path.exists(video_path):
                return {"error": "Video generation failed - no output file created"}
                
            print(f"✓ Video generated successfully: {video_path}")
            
        except Exception as e:
            print(f"Video generation failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Video generation failed: {str(e)}"}

        # Upload video to R2
        try:
            video_filename = f"{job_id}.mp4"
            video_url = upload_video(video_path, video_filename)
            
            # Get video metadata
            video_size_mb = os.path.getsize(video_path) / 1024 / 1024
            
            job_output = {
                "video_url": video_url,
                "video_size_mb": round(video_size_mb, 2),
                "parameters": {
                    "prompt": prompt,
                    "size": size,
                    "num_frames": num_frames,
                    "guidance_scale": guidance_scale,
                    "num_inference_steps": num_inference_steps,
                    "seed": seed,
                    "fps": fps,
                    "use_prompt_extend": use_prompt_extend
                }
            }
            
            print(f"✓ Job completed successfully: {video_filename} ({video_size_mb:.2f} MB)")
            
        except Exception as e:
            print(f"Video upload failed: {e}")
            return {"error": f"Video upload failed: {str(e)}"}

        # Clean up temporary files
        try:
            # Remove downloaded input files
            rp_cleanup.clean(['input_objects'])
            
            # Remove generated video file
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
                print("✓ Temporary files cleaned up")
                
        except Exception as cleanup_error:
            print(f"Warning: Cleanup failed: {cleanup_error}")

        # ===== ENHANCED MEMORY CLEANUP =====
        # Force memory cleanup to prevent accumulation between jobs
        try:
            import gc
            import torch
            
            # Clear any remaining variables
            locals_to_clean = ['video_path', 'image_path', 'video_url']
            for var_name in locals_to_clean:
                if var_name in locals() and locals()[var_name] is not None:
                    del locals()[var_name]
            
            # PyTorch GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for GPU operations to complete
                
                # Get memory stats for monitoring
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_cached = torch.cuda.memory_reserved() / 1024**3      # GB
                print(f"Post-job GPU Memory - Allocated: {memory_allocated:.2f}GB, Cached: {memory_cached:.2f}GB")
            
            # Force garbage collection
            gc.collect()
            
            print(f"✓ Memory cleanup completed for job {job_id}")
            
        except Exception as cleanup_error:
            print(f"Warning: Post-job memory cleanup failed: {cleanup_error}")
        # ===== END ENHANCED MEMORY CLEANUP =====

        return job_output

    except Exception as e:
        print(f"Error processing job {job.get('id', 'unknown')}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"Internal error: {str(e)}"}


if __name__ == "__main__":
    # Initialize model
    print("Initializing Wan 2.2 TI2V-5B model...")
    try:
        MODEL = predict.Predictor(model_dir=model_dir)
        MODEL.setup()
        print("✓ Model initialized successfully")
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # Start RunPod serverless
    print("Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": run}) 