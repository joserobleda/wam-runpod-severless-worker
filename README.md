# 🧠 Text-to-Video Serverless Deployment (RunPod)

This repository provides a serverless deployment template for **text-to-video generation** using RunPod. It allows you to run CogVideoX-5b generative video model from a text prompt via a simple API—fully hosted on RunPod's secure, serverless GPU environment with automatic Cloudflare R2 storage upload.

---

## 🚀 Features

* 🔄 **Text-to-Video** generation using CogVideoX-5b model
* 🌐 **API endpoint** support for easy integration
* ⚙️ **RunPod serverless template** with `rp_handler.py` and custom Dockerfile
* ☁️ **Automatic R2 Upload**: Videos are automatically uploaded to Cloudflare R2 storage
* 💸 **Low cost & scalable**—only pay when your model is running
* 🎬 **High Quality**: State-of-the-art video generation with configurable parameters

---

## 🛠️ Requirements

* RunPod account: https://www.runpod.io/
* Cloudflare R2 storage account for video uploads
* Docker knowledge (basic)
* GPU with 24GB+ VRAM (RTX 4090, A100, or similar)
* This repo cloned locally or on GitHub

---

## ⚙️ Deployment Instructions

### 1. Clone this repo

```bash
git clone https://github.com/your-username/cogvideox-runpod-template.git
cd cogvideox-runpod-template
```

### 2. Set up your Cloudflare R2 Storage

Configure these environment variables in RunPod:

```bash
# Required for R2 Storage
BUCKET_ENDPOINT_URL=https://your-account-id.r2.cloudflarestorage.com/your-bucket-name
BUCKET_ACCESS_KEY_ID=your-r2-access-key
BUCKET_SECRET_ACCESS_KEY=your-r2-secret-key
BUCKET_NAME=your-bucket-name  # Optional if included in endpoint URL
```

### 3. Create a Serverless Endpoint on RunPod

1. Log in to RunPod
2. Go to **"Serverless > Community Templates"**
3. Click **"Create Endpoint"** and select **"Custom Template"**
4. Upload this repository (or connect your GitHub repo)
5. Set the environment variables for R2 storage
6. RunPod will automatically detect `rp_handler.py` and the `Dockerfile`

### 4. Test the Endpoint

Use the sample `test_input.json` file or send a POST request like this:

```json
{
    "input": {
        "mode": "txt2video",
        "prompt": "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance",
        "negative_prompt": "",
        "num_frames": 48,
        "guidance_scale": 6,
        "aspect_ratio": "1:1",
        "num_inference_steps": 50,
        "max_sequence_length": 226,
        "fps": 8
    }
}
```

---

## 📥 Sample API Call

```bash
curl -X POST https://api.runpod.ai/v2/YOUR-ENDPOINT-ID/run \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d @test_input.json
```

---

## 📋 API Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `mode` | string | No | "txt2video" | Generation mode (currently only supports txt2video) |
| `prompt` | string | Yes | - | Text description of the video to generate |
| `negative_prompt` | string | No | "" | What to avoid in the generation |
| `num_frames` | integer | No | 48 | Number of frames (16-120) |
| `guidance_scale` | float | No | 6.0 | How closely to follow the prompt (1.0-20.0) |
| `aspect_ratio` | string | No | "16:9" | Video aspect ratio (informational) |
| `num_inference_steps` | integer | No | 50 | Denoising steps for quality (10-100) |
| `max_sequence_length` | integer | No | 226 | Maximum text sequence length (128-512) |
| `fps` | integer | No | 8 | Frames per second (6-24) |
| `seed` | integer | No | None | Random seed for reproducible results |

---

## 🎯 Example Responses

### Successful Response
```json
{
    "id": "job-12345-abcde",
    "status": "COMPLETED",
    "output": {
        "video_url": "https://your-r2-endpoint.com/your-bucket/job-12345-abcde.mp4",
        "parameters": {
            "mode": "txt2video",
            "prompt": "A cat walking in a garden",
            "negative_prompt": "",
            "num_frames": 24,
            "guidance_scale": 8.5,
            "aspect_ratio": "16:9",
            "num_inference_steps": 25,
            "max_sequence_length": 226,
            "fps": 12,
            "seed": null
        }
    }
}
```

### Error Response
```json
{
    "id": "job-12345-abcde",
    "status": "FAILED",
    "error": "Video generation failed: CUDA out of memory"
}
```

---

## 🏃‍♂️ Local Testing

You can test video generation locally using the included `generate_video.py` script:

```bash
# Simple generation
python generate_video.py --prompt "A cat walking in a garden" --output "my_video.mp4"

# Advanced generation with custom parameters
python generate_video.py \
    --prompt "A dragon flying through clouds" \
    --num-frames 60 \
    --guidance-scale 7.0 \
    --num-inference-steps 40 \
    --fps 12 \
    --output "dragon_video.mp4"

# Load parameters from config file
python generate_video.py --config test_input.json --output "config_video.mp4"
```

---

## 📌 Notes

* **GPU Requirements**: You need a GPU with at least 24 GB VRAM for CogVideoX-5b
* **Model Download**: The model (~15GB) will be downloaded automatically on first run
* **Generation Time**: Varies by prompt and frame count (typically 5-15 minutes)
* **R2 Storage**: Videos are automatically uploaded to your configured R2 bucket
* **File Cleanup**: Temporary files are cleaned up after successful upload

---

## 🎨 Model Information

- **Model**: CogVideoX-5b from THUDM
- **Framework**: Diffusers + Transformers  
- **Precision**: Float16 for memory efficiency
- **Max Frames**: Up to 120 frames
- **Quality**: State-of-the-art video generation

---

## ☁️ R2 Storage Integration

This template includes automatic upload to Cloudflare R2 storage:

1. **Generated videos** are automatically uploaded to your R2 bucket
2. **Public URLs** are returned for easy access and sharing
3. **Temporary files** are cleaned up after upload
4. **Error handling** for upload failures with fallback responses

---

## 📄 License

MIT License. Feel free to modify and use for commercial or personal projects.