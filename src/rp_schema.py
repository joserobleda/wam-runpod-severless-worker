INPUT_SCHEMA = {
    "prompt": {
        "type": str,
        "required": True,
        "description": "Text prompt describing the video to generate"
    },
    "image": {
        "type": str,
        "required": False,
        "default": None,
        "description": "URL of input image for image-to-video generation (optional, for text-to-video leave empty)"
    },
    "size": {
        "type": str,
        "required": False,
        "default": "1280*704",
        "description": "Video resolution in format 'width*height'. Default is 1280*704 (720P). Also supports 704*1280"
    },
    "num_frames": {
        "type": int,
        "required": False,
        "default": 121,
        "min": 24,
        "max": 240,
        "description": "Number of frames to generate (121 = 5 seconds at 24fps)"
    },
    "guidance_scale": {
        "type": float,
        "required": False,
        "default": 5.0,
        "min": 1.0,
        "max": 20.0,
        "description": "Guidance scale for generation quality vs diversity trade-off"
    },
    "num_inference_steps": {
        "type": int,
        "required": False,
        "default": 50,
        "min": 10,
        "max": 100,
        "description": "Number of denoising steps (more steps = higher quality but slower)"
    },
    "seed": {
        "type": int,
        "required": False,
        "default": None,
        "description": "Random seed for reproducible generation (optional)"
    },
    "fps": {
        "type": int,
        "required": False,
        "default": 24,
        "min": 12,
        "max": 30,
        "description": "Frames per second for the output video"
    },
    "negative_prompt": {
        "type": str,
        "required": False,
        "default": None,
        "description": "What to avoid in the generation (optional, uses Wan 2.2 defaults if not provided)"
    },
    "use_prompt_extend": {
        "type": bool,
        "required": False,
        "default": False,
        "description": "Whether to extend short prompts with quality descriptors"
    }
} 