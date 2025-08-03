# Wan2.2 RunPod Serverless Deployment Checklist ✅

## 📋 Pre-Deployment Validation

### 1. Run Validation Script
```bash
python validate_deployment.py
```
**Expected Output:** All tests should pass with green checkmarks ✅

### 2. Required Files Verification
Ensure all these files exist in your deployment directory:

**Core Files:**
- ✅ `src/handler.py` - Main RunPod handler
- ✅ `src/predict.py` - Video generation logic  
- ✅ `src/utils.py` - Utility functions
- ✅ `src/__init__.py` - Package initialization
- ✅ `builder/download_model.py` - Model download script
- ✅ `builder/requirements.txt` - Python dependencies
- ✅ `Dockerfile` - Container definition

**Configuration Files:**
- ✅ `test_input.json` - Simple test case
- ✅ `test_input_complex.json` - Complex test case
- ✅ `README.md` - Comprehensive documentation
- ✅ `deploy.sh` - Deployment script
- ✅ `test_endpoint.sh` - Endpoint testing script

## 🛠️ Build & Deploy Process

### 1. Build Docker Image
```bash
# Make scripts executable
chmod +x deploy.sh test_endpoint.sh

# Build and push image
./deploy.sh -p YOUR_DOCKERHUB_USERNAME
```

### 2. RunPod Configuration
**Required Settings:**
- **Docker Image:** `docker.io/YOUR_USERNAME/wan22-t2v-diffusers:latest`
- **GPU:** NVIDIA A100 (40GB or 80GB)
- **Network Volume:** Create and attach (minimum 100GB)
- **Min Workers:** 0
- **Max Workers:** 1+ (based on expected load)

**Environment Variables (Optional):**
```
HF_HUB_CACHE=/runpod-volume/.cache/huggingface
```

### 3. Test Deployment
```bash
# Test with simple input
./test_endpoint.sh YOUR_ENDPOINT_ID YOUR_API_KEY

# Test with complex input  
./test_endpoint.sh -f test_input_complex.json YOUR_ENDPOINT_ID YOUR_API_KEY
```

## 🔍 Critical Implementation Details

### Model Configuration ✅
- **Model ID:** `Wan-AI/Wan2.2-T2V-A14B-Diffusers`
- **Model Path:** `/runpod-volume/model`
- **Output Path:** `/runpod-volume/outputs`
- **Pipeline:** `WanPipeline` from diffusers

### Hardware Requirements ✅
- **Minimum GPU:** A100 40GB VRAM
- **Recommended GPU:** A100 80GB VRAM
- **Storage:** 100GB+ for model and outputs
- **RAM:** 16GB+ system memory

### API Interface ✅
**Request Format:**
```json
{
  "input": {
    "prompt": "Your video description",
    "negative_prompt": "What to avoid",
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

**Response Format:**
```json
{
  "status": "success",
  "video_path": "/runpod-volume/outputs/wan22_video_1234567890_abcd1234.mp4",
  "metadata": {
    "generation_time_seconds": 120.5,
    "num_frames": 81,
    "parameters": { ... },
    "timing": { ... },
    "memory_usage_after": { ... }
  }
}
```

## ✅ Implementation Fixes Applied

### Critical Fixes Made:
1. **✅ Corrected Model Loading**
   - Removed manual VAE loading
   - Using direct `WanPipeline.from_pretrained()`
   - Proper torch_dtype handling

2. **✅ Fixed Imports**
   - Removed unnecessary transformers imports
   - Added `export_to_video` from diffusers.utils
   - Streamlined dependency management

3. **✅ Parameter Validation**
   - Comprehensive input validation
   - Range checking with fallbacks
   - Proper error handling

4. **✅ Memory Management**
   - GPU memory monitoring
   - Automatic cleanup on errors
   - Resource optimization techniques

5. **✅ Error Handling**
   - Detailed error messages
   - Graceful failure modes
   - Comprehensive logging

## 🚀 Performance Expectations

### Generation Times:
- **720P Video (81 frames):** ~2-3 minutes on A100 80GB
- **480P Video (81 frames):** ~1-2 minutes on A100 80GB
- **Model Loading:** ~30-60 seconds (first request only)

### Memory Usage:
- **A100 40GB:** Sufficient for 720P generation
- **A100 80GB:** Optimal performance and headroom
- **Peak VRAM:** ~35-45GB during generation

### Cost Estimation:
- **A100 40GB:** ~$1.69/hour → ~$0.08 per video
- **A100 80GB:** ~$2.29/hour → ~$0.11 per video
- **Network Volume:** ~$0.05-0.07/GB/month

## 🔧 Troubleshooting Guide

### Common Issues & Solutions:

**1. Model Download Fails**
```bash
# Check network connectivity and retry
# Verify model ID is correct: Wan-AI/Wan2.2-T2V-A14B-Diffusers
```

**2. Out of Memory Errors**
```bash
# Use A100 80GB instead of 40GB
# Reduce resolution: 480P instead of 720P
# Check if other processes are using GPU memory
```

**3. Container Won't Start**
```bash
# Verify Docker image URL is correct
# Check RunPod logs for specific errors
# Ensure network volume is attached
```

**4. Generation Fails**
```bash
# Check prompt length and content
# Verify all parameters are within valid ranges
# Check GPU availability and memory
```

**5. Slow Performance**
```bash
# Verify A100 GPU allocation
# Check if CPU offloading is working
# Monitor memory usage during generation
```

## 📈 Monitoring & Health Checks

### Built-in Monitoring:
- **Health Check Endpoint:** Automatic container health monitoring
- **Memory Tracking:** Real-time GPU memory usage
- **Performance Metrics:** Generation time tracking
- **Error Logging:** Comprehensive error capture

### Key Metrics to Watch:
- **GPU Memory Usage:** Should not exceed 90% of available VRAM
- **Generation Time:** Should be consistent with expectations
- **Success Rate:** Should be >95% for valid inputs
- **Model Loading Time:** Should only occur on first request

## ✅ Final Validation Checklist

Before deploying to production:

- [ ] All validation tests pass (`python validate_deployment.py`)
- [ ] Docker image builds successfully 
- [ ] Image pushed to DockerHub
- [ ] RunPod endpoint configured correctly
- [ ] Network volume attached and accessible
- [ ] Test requests complete successfully
- [ ] Performance meets expectations
- [ ] Error handling works properly
- [ ] Monitoring and logging functional

## 🎯 Success Criteria

**Deployment is successful when:**
1. ✅ Container starts without errors
2. ✅ Model downloads and loads correctly
3. ✅ Health check returns "healthy" status
4. ✅ Test video generation completes in expected time
5. ✅ Generated video file is accessible and playable
6. ✅ Memory usage stays within limits
7. ✅ API responses match expected format

---

**🎉 Ready for Production!**

Your Wan2.2 RunPod serverless deployment is now ready for production use. The implementation follows best practices for scalability, reliability, and performance. 