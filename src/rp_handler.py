import io
import os
import sys
import argparse
import base64
import tempfile
import shutil
from pathlib import Path
from typing import Optional
import torch  # Added for debug logging
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
from predict import Predictor

# Initialize the model when the worker starts.
# This will run the setup() method and prepare the pipeline.
model = Predictor()
model.setup()


def upload_to_r2(file_path, job_id):
    """Uploads a file to a Cloudflare R2 bucket."""

    endpoint_url = os.environ.get('BUCKET_ENDPOINT_URL')
    access_key_id = os.environ.get('BUCKET_ACCESS_KEY_ID')
    secret_access_key = os.environ.get('BUCKET_SECRET_ACCESS_KEY')

    required_vars = {
        'BUCKET_ENDPOINT_URL': endpoint_url,
        'BUCKET_ACCESS_KEY_ID': access_key_id,
        'BUCKET_SECRET_ACCESS_KEY': secret_access_key,
    }

    missing_vars = [key for key, value in required_vars.items() if not value]
    if missing_vars:
        print(f"❌ R2 Upload Failed: Missing environment variables: {', '.join(missing_vars)}")
        return None

    try:
        # If the URL contains a bucket name, extract it
        if '/' in endpoint_url.split('://', 1)[1]:
            # URL format: https://account-id.r2.cloudflarestorage.com/bucket-name
            base_url, bucket_name = endpoint_url.rsplit('/', 1)
            actual_endpoint = base_url
        else:
            # URL format: https://account-id.r2.cloudflarestorage.com
            actual_endpoint = endpoint_url
            bucket_name = os.environ.get('BUCKET_NAME')
            if not bucket_name:
                print("❌ R2 Upload Failed: BUCKET_NAME environment variable is required when bucket is not in the endpoint URL.")
                return None

        s3_client = boto3.client(
            's3',
            endpoint_url=actual_endpoint,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name='auto'
        )

        file_key = f"{job_id}.mp4"

        s3_client.upload_file(file_path, bucket_name, file_key)

        # Construct the public URL from the endpoint and bucket name
        url = f"{actual_endpoint}/{bucket_name}/{file_key}"
        print(f"✅ Successfully uploaded to R2: {url}")
        return url
    except ClientError as e:
        print(f"❌ R2 Upload Failed: {e}")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred during upload: {e}")
        return None


def handler(job):
    """
    The handler function for the RunPod serverless worker.
    """
    job_input = job['input']

    # Extract parameters from the job input with defaults
    prompt = job_input.get('prompt', "a photo of an astronaut riding a horse on mars")
    negative_prompt = job_input.get('negative_prompt', "blurry, low quality, static, poorly drawn, deformed")
    size = job_input.get('size', '512x512')
    num_frames = job_input.get('num_frames', 24)
    num_inference_steps = job_input.get('num_inference_steps', 25)
    guidance_scale = job_input.get('guidance_scale', 8.5)
    fps = job_input.get('fps', 12)
    seed = job_input.get('seed', None)

    # Generate the video
    video_path = model.predict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        size=size,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        fps=fps,
        seed=seed,
    )

    # Upload the generated video to R2
    video_url = upload_to_r2(video_path, job['id'])

    # Clean up the temporary file
    rp_cleanup.clean([os.path.dirname(video_path)])

    if not video_url:
        return {"error": "Video generation succeeded, but upload failed."}
        
    return {"video_url": video_url}

# Start the serverless worker
runpod.serverless.start({"handler": handler}) 