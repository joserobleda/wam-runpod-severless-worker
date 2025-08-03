# Wan-AI/Wan2.2-T2V-A14B-Diffusers RunPod Serverless Deployment

This repository provides a complete solution for deploying the [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) text-to-video generation model on RunPod's serverless platform.

## 🌟 Features

- **State-of-the-art Video Generation**: Deploy Wan2.2, a cutting-edge text-to-video model with 27B parameters (14B active)
- **Scalable Serverless Architecture**: Automatic scaling with pay-per-second billing
- **Optimized Performance**: Memory-efficient implementation with GPU optimizations
- **Comprehensive API**: Full-featured REST API with detailed error handling
- **Production Ready**: Health checks, logging, and proper resource management
- **Easy Deployment**: One-click deployment scripts and detailed documentation

## 🏗️ Architecture

The deployment uses a modular architecture:

```
wan-video-runpod-serverless/
├── builder/                 # Model downloading and setup
│   ├── requirements.txt     # Python dependencies
│   └── download_model.py    # Model download script
├── src/                     # Core application logic
│   ├── handler.py          # RunPod serverless handler
│   ├── predict.py          # Video generation logic
│   ├── utils.py            # Utility functions
│   └── __init__.py
├── Dockerfile              # Container definition
├── deploy.sh               # Deployment script
├── test_endpoint.sh        # Endpoint testing script
├── test_input.json         # Simple test input
├── test_input_complex.json # Complex test input
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed
- [RunPod](https://runpod.io) account with API key
- Docker Hub account (for image hosting)

### 1. Clone and Build

```bash
# Clone or download this repository
git clone <your-repo-url>
cd wan-video-runpod-serverless

# Build and push Docker image
./deploy.sh -p YOUR_DOCKERHUB_USERNAME
```

### 2. Deploy on RunPod

1. Go to [RunPod Console](https://www.runpod.io/console/serverless)
2. Click "New Endpoint"
3. Select "Custom Source" → "Docker Image"
4. Enter your image: `docker.io/YOUR_USERNAME/wan22-t2v-diffusers:latest`
5. Configure:
   - **GPU**: NVIDIA A100 (40GB or 80GB recommended)
   - **Network Volume**: Create and attach for model persistence
   - **Worker Config**: Min: 0, Max: 1+ (based on expected load)

### 3. Test the Endpoint

```bash
# Test with simple input
./test_endpoint.sh YOUR_ENDPOINT_ID YOUR_API_KEY

# Test with complex input
./test_endpoint.sh -f test_input_complex.json YOUR_ENDPOINT_ID YOUR_API_KEY
```

## 📋 Detailed Setup

### System Requirements

| Component | Requirement | Notes |
|-----------|-------------|-------|
| GPU | NVIDIA A100 40GB+ | Required for model inference |
| Storage | 20GB+ | For model files and generated videos |
| RAM | 16GB+ | For efficient processing |
| Network | High bandwidth | For model download (~50GB) |

### Environment Variables (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_HUB_CACHE` | `/runpod-volume/.cache/huggingface` | Hugging Face cache directory |
| `MODEL_DIR` | `/runpod-volume/model` | Model storage directory |
| `OUTPUT_DIR` | `/runpod-volume/outputs` | Generated video output directory |

### Deployment Script Options

```bash
./deploy.sh [OPTIONS] USERNAME

Options:
  -t, --tag TAG        Image tag (default: latest)
  -p, --push           Push to registry after building
  -f, --force          Force rebuild without cache
  -h, --help           Show help message

Examples:
  ./deploy.sh myusername                    # Build image only
  ./deploy.sh -p myusername                 # Build and push
  ./deploy.sh -t v1.0.0 -p myusername       # Custom tag
  ./deploy.sh -f -p myusername              # Force rebuild
```

## 🔧 API Reference

### Request Format

```json
{
  "input": {
    "prompt": "Your video description here",
    "negative_prompt": "What to avoid (optional)",
    "num_frames": 81,
    "guidance_scale": 6.0,
    "num_inference_steps": 40,
    "width": 720,
    "height": 480,
    "fps": 16,
    "seed": 42
  }
}
```

### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `prompt` | string | **required** | - | Text description for video generation |
| `negative_prompt` | string | `""` | - | What to avoid in the generated video |
| `num_frames` | integer | `81` | 1-200 | Number of video frames to generate |
| `guidance_scale` | float | `5.0` | 1.0-20.0 | How closely to follow the prompt |
| `num_inference_steps` | integer | `40` | 10-100 | Number of denoising steps |
| `width` | integer | `720` | 256-1920 | Video width in pixels |
| `height` | integer | `480` | 256-1080 | Video height in pixels |
| `fps` | integer | `16` | 1-60 | Frames per second (metadata only) |
| `seed` | integer | `null` | - | Random seed for reproducibility |

### Response Format

#### Success Response

```json
{
  "status": "success",
  "video_path": "/runpod-volume/outputs/wan22_video_1234567890_abcd1234.mp4",
  "metadata": {
    "prompt": "Your prompt here",
    "num_frames": 81,
    "generation_time_seconds": 120.5,
    "generation_time_formatted": "2.0m",
    "parameters": { ... },
    "timing": {
      "total_request_time_seconds": 125.8,
      "generation_time_seconds": 120.5,
      "save_time_seconds": 2.1
    },
    "memory_usage_after": {
      "allocated_gb": 35.2,
      "reserved_gb": 40.0,
      "total_gb": 80.0,
      "free_gb": 40.0
    },
    "video_info": {
      "path": "/runpod-volume/outputs/wan22_video_1234567890_abcd1234.mp4",
      "filename": "wan22_video_1234567890_abcd1234.mp4",
      "size_bytes": 15728640
    }
  },
  "timestamp": 1672531200
}
```

#### Error Response

```json
{
  "status": "error",
  "error_type": "validation_error",
  "message": "Prompt cannot be empty",
  "timestamp": 1672531200
}
```

### Error Types

| Error Type | Description |
|------------|-------------|
| `validation_error` | Invalid input parameters |
| `model_error` | Model loading or setup issues |
| `processing_error` | Video generation failures |

## 🧪 Testing

### Local Testing

```bash
# Test endpoint connectivity
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d @test_input.json

# Using the test script (recommended)
./test_endpoint.sh YOUR_ENDPOINT_ID YOUR_API_KEY
```

### Test Files

- `test_input.json`: Simple test with basic parameters
- `test_input_complex.json`: Complex test with detailed parameters

### Performance Benchmarks

Expected performance on NVIDIA A100:

| Resolution | Frames | Inference Steps | Generation Time | Memory Usage |
|------------|--------|-----------------|-----------------|--------------|
| 720x480 | 81 | 40 | ~2-3 minutes | ~35GB |
| 1280x720 | 81 | 40 | ~4-6 minutes | ~45GB |
| 720x480 | 81 | 50 | ~3-4 minutes | ~35GB |

## 🔍 Monitoring and Debugging

### Health Check

The container includes a health check endpoint:

```bash
# Check container health
docker exec CONTAINER_ID /health_check.sh
```

### Logs

View container logs for debugging:

```bash
# RunPod console logs
# Check the "Logs" tab in your endpoint dashboard

# Local testing logs
docker logs CONTAINER_ID
```

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Out of Memory | Insufficient GPU VRAM | Use A100 80GB or reduce resolution |
| Model Download Fails | Network issues | Check internet connection, retry |
| Container Won't Start | Docker image issues | Rebuild with `--force` flag |
| Slow Generation | GPU underutilization | Check GPU allocation in RunPod |

## 🛠️ Customization

### Modifying Parameters

Edit `src/utils.py` to change default parameters:

```python
# In validate_input function
params = {
    'num_frames': int(input_data.get('num_frames', 81)),  # Change default here
    'guidance_scale': float(input_data.get('guidance_scale', 5.0)),
    # ...
}
```

### Adding Features

1. **Custom Preprocessing**: Modify `src/predict.py`
2. **Additional Outputs**: Update `src/utils.py`
3. **New Endpoints**: Extend `src/handler.py`

### Performance Optimization

1. **Enable XFormers**: Automatically enabled if available
2. **Model Precision**: Uses bfloat16 by default for efficiency
3. **Memory Offloading**: CPU offloading enabled for memory efficiency

## 📊 Cost Estimation

RunPod pricing (approximate):

| GPU | Price/hour | Generation Time | Cost per Video |
|-----|------------|-----------------|----------------|
| A100 40GB | $1.69/hour | 3 minutes | ~$0.08 |
| A100 80GB | $2.29/hour | 3 minutes | ~$0.11 |

Additional costs:
- Network Volume: $0.05-0.07/GB/month
- Idle time: $0 (serverless scales to zero)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This deployment code is provided under the Apache 2.0 License. The Wan2.2 model is also licensed under Apache 2.0.

## 🆘 Support

- **RunPod Documentation**: [docs.runpod.io](https://docs.runpod.io)
- **Wan2.2 Model**: [Hugging Face Model Card](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers)
- **Issues**: Open an issue in this repository

## 🙏 Acknowledgments

- **Wan-AI Team**: For the amazing Wan2.2 model
- **RunPod**: For the serverless infrastructure
- **Hugging Face**: For model hosting and diffusers library
- **Community**: For feedback and contributions

---

**Happy video generating!** 🎬✨ 