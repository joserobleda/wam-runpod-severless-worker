INPUT_SCHEMA = {
    "mode": {
        "type": str,
        "required": False,
        "default": "txt2video",
        "description": "Generation mode (txt2video for text-to-video generation)"
    },
    "prompt": {
        "type": str,
        "required": True,
        "description": "Text prompt describing the video to generate"
    },
    "negative_prompt": {
        "type": str,
        "required": False,
        "default": "",
        "description": "What to avoid in the generation (optional, uses CogVideoX defaults if not provided)"
    },
    "num_frames": {
        "type": int,
        "required": False,
        "default": 48,
        "min": 16,
        "max": 120,
        "description": "Number of frames to generate (48 = ~2 seconds at 24fps)"
    },
    "guidance_scale": {
        "type": float,
        "required": False,
        "default": 6.0,
        "min": 1.0,
        "max": 20.0,
        "description": "Guidance scale for generation quality vs diversity trade-off"
    },
    "aspect_ratio": {
        "type": str,
        "required": False,
        "default": "16:9",
        "description": "Video aspect ratio (16:9, 1:1, 9:16, etc.)"
    },
    "num_inference_steps": {
        "type": int,
        "required": False,
        "default": 50,
        "min": 10,
        "max": 100,
        "description": "Number of denoising steps (more steps = higher quality but slower)"
    },
    "max_sequence_length": {
        "type": int,
        "required": False,
        "default": 226,
        "min": 128,
        "max": 512,
        "description": "Maximum sequence length for text encoding"
    },
    "fps": {
        "type": int,
        "required": False,
        "default": 8,
        "min": 6,
        "max": 24,
        "description": "Frames per second for the output video"
    },
    "seed": {
        "type": int,
        "required": False,
        "default": None,
        "description": "Random seed for reproducible generation (optional)"
    }
} 