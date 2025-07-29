# Wan 2.2 TI2V-5B RunPod Serverless Worker

A RunPod serverless worker for generating videos using the [Wan 2.2 TI2V-5B model](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B). This service supports both text-to-video and image-to-video generation at 720P resolution with 24fps.

## Features

- 🎬 **Text-to-Video Generation**: Generate videos from text prompts
- 🖼️ **Image-to-Video Generation**: Animate static images with text guidance
- 🚀 **High Performance**: Optimized for single consumer-grade GPUs (RTX 4090)
- ☁️ **Cloud Storage**: Automatic upload to Cloudflare R2 storage
- 🔧 **Memory Optimized**: Efficient GPU memory management for serverless deployment
- 📏 **720P Quality**: Generate videos at 1280x704 or 704x1280 resolution
- ⚡ **Fast Generation**: One of the fastest 720P@24fps models available

## Model Information

- **Model**: Wan 2.2 TI2V-5B (5 billion parameters)
- **Resolution**: Supports 720P (1280x704 or 704x1280)
- **Frame Rate**: 24 FPS
- **Max Duration**: Up to 10 seconds (240 frames)
- **Compression**: High-compression VAE with 16×16×4 ratio

## Requirements

- GPU: 24GB+ VRAM (RTX 4090 or better)
- Storage: ~15GB for model weights
- Memory: 32GB+ RAM recommended

## Quick Start

### 1. Build and Deploy

```bash
# Clone the repository
git clone <repository-url>
cd wam-runpod-severless-worker

# Build the Docker image
docker build -t wan22-ti2v-5b .

# Deploy to RunPod (replace with your endpoint)
# Upload to RunPod container registry and deploy
```

### 2. Environment Variables

Set these environment variables in your RunPod endpoint:

```bash
# Required for R2 Storage
BUCKET_ENDPOINT_URL=https://your-account-id.r2.cloudflarestorage.com/your-bucket-name
BUCKET_ACCESS_KEY_ID=your-r2-access-key
BUCKET_SECRET_ACCESS_KEY=your-r2-secret-key
BUCKET_NAME=wam-videos  # Optional, defaults to 'wam-videos'

# Worker Configuration (auto-set by RunPod)
WORKER_MODEL_DIR=/app/model
WORKER_USE_CUDA=True
```

### 3. Test the Service

Send a POST request to your RunPod endpoint:

```bash
curl -X POST https://api.runpod.ai/v2/your-endpoint-id/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "input": {
      "prompt": "A majestic golden retriever running through a sunlit meadow with wildflowers, slow motion, cinematic quality"
    }
  }'
```

## API Reference

### Input Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | Yes | - | Text prompt describing the video to generate |
| `image` | string | No | null | URL of input image for image-to-video generation |
| `size` | string | No | "1280*704" | Video resolution (1280*704 or 704*1280) |
| `num_frames` | integer | No | 120 | Number of frames (24-240, 24fps) |
| `guidance_scale` | float | No | 7.0 | Guidance scale (1.0-20.0) |
| `num_inference_steps` | integer | No | 50 | Denoising steps (10-100) |
| `seed` | integer | No | null | Random seed for reproducible results |
| `fps` | integer | No | 24 | Frames per second (12-30) |
| `use_prompt_extend` | boolean | No | false | Automatically enhance the prompt |

### Example Requests

#### Simple Text-to-Video
```json
{
  "input": {
    "prompt": "A cat playing with a ball of yarn in a cozy living room"
  }
}
```

#### Advanced Text-to-Video
```json
{
  "input": {
    "prompt": "Epic battle scene with dragons flying over a medieval castle, cinematic quality, dramatic lighting",
    "size": "1280*704",
    "num_frames": 180,
    "guidance_scale": 8.0,
    "num_inference_steps": 60,
    "seed": 12345,
    "fps": 24,
    "use_prompt_extend": true
  }
}
```

#### Image-to-Video
```json
{
  "input": {
    "prompt": "The person in the image starts walking forward with a confident smile",
    "image": "https://example.com/your-image.jpg",
    "size": "1280*704",
    "num_frames": 120,
    "guidance_scale": 7.0
  }
}
```

### Response Format

```json
{
  "delayTime": 1250,
  "executionTime": 45000,
  "id": "job-id-12345",
  "output": {
    "video_url": "https://your-r2-bucket.com/job-id-12345.mp4",
    "video_size_mb": 12.5,
    "parameters": {
      "prompt": "Your input prompt",
      "size": "1280*704",
      "num_frames": 120,
      "guidance_scale": 7.0,
      "num_inference_steps": 50,
      "seed": null,
      "fps": 24,
      "use_prompt_extend": false
    }
  },
  "status": "COMPLETED"
}
```

## Performance

### Generation Times (RTX 4090)

| Configuration | Time | Notes |
|---------------|------|-------|
| 5s @ 720P (120 frames) | ~8-12 min | Default settings |
| 3s @ 720P (72 frames) | ~5-8 min | Faster generation |
| 7s @ 720P (168 frames) | ~12-18 min | Longer videos |

### Memory Usage

- **Model Loading**: ~12GB VRAM
- **Peak Generation**: ~20GB VRAM
- **Idle**: ~8GB VRAM

## Tips for Better Results

### Prompt Engineering
- Use descriptive, detailed prompts
- Include cinematic terms: "cinematic quality", "professional lighting", "smooth motion"
- Specify camera movements: "slow motion", "close-up", "wide shot"
- Add style descriptors: "dramatic", "serene", "epic", "realistic"

### Image-to-Video Tips
- Use high-quality input images (1280x704 recommended)
- Ensure good lighting and clear subjects in input images
- Prompts should describe the motion you want to see

### Performance Optimization
- Use fewer inference steps (20-30) for faster generation
- Reduce frame count for quicker results
- Enable prompt extension for better quality with simple prompts

## Troubleshooting

### Common Issues

**Model Loading Fails**
- Ensure sufficient GPU memory (24GB+)
- Check model files were downloaded correctly during build

**Generation Timeout**
- Reduce `num_frames` or `num_inference_steps`
- Check GPU memory availability

**Upload Failures**
- Verify R2 credentials are correct
- Check bucket permissions
- Ensure sufficient storage space

**Out of Memory**
- Reduce batch size in generation
- Lower resolution if needed
- Clear cache between generations

### Monitoring

The worker provides detailed logging:
- Model loading status
- GPU memory usage
- Generation progress
- Upload confirmation
- Error details

## Development

### Local Development

```bash
# Clone and setup
git clone <repository-url>
cd wam-runpod-severless-worker

# Install dependencies
pip install -r builder/requirements.txt

# Download model manually (if needed)
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./model/wan2.2-ti2v-5b

# Test locally
python src/rp_handler.py
```

### Custom Model Path

To use a different model location:
```bash
export WORKER_MODEL_DIR=/path/to/your/models
python src/rp_handler.py
```

## License

This project uses the Wan 2.2 model under the Apache 2.0 License. See the [original repository](https://github.com/Wan-Video/Wan2.2) for details.

## Acknowledgments

- [Wan-AI Team](https://huggingface.co/Wan-AI) for the amazing Wan 2.2 model
- [RunPod](https://runpod.io) for the serverless infrastructure
- [Diffusers](https://huggingface.co/docs/diffusers) for the pipeline implementation

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review RunPod logs for detailed error messages
3. Ensure all environment variables are set correctly
4. Verify GPU requirements are met