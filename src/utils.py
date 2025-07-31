import base64
import os
import tempfile
import logging
from pathlib import Path
from typing import Union, Optional

def encode_video_to_base64(video_path: str) -> str:
    """
    Encode a video file to base64 string.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        str: Base64 encoded video data
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    with open(video_path, "rb") as video_file:
        video_data = video_file.read()
        base64_data = base64.b64encode(video_data).decode('utf-8')
        
    return base64_data


def decode_base64_to_video(base64_data: str, output_path: str = None) -> str:
    """
    Decode base64 string to video file.
    
    Args:
        base64_data (str): Base64 encoded video data
        output_path (str, optional): Path to save the video. If None, creates temp file.
        
    Returns:
        str: Path to the decoded video file
    """
    if output_path is None:
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        output_path = temp_file.name
        temp_file.close()
    
    video_data = base64.b64decode(base64_data)
    
    with open(output_path, "wb") as video_file:
        video_file.write(video_data)
        
    return output_path


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64 string.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Base64 encoded image data
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        base64_data = base64.b64encode(image_data).decode('utf-8')
        
    return base64_data


def decode_base64_to_image(base64_data: str, output_path: str = None) -> str:
    """
    Decode base64 string to image file.
    
    Args:
        base64_data (str): Base64 encoded image data
        output_path (str, optional): Path to save the image. If None, creates temp file.
        
    Returns:
        str: Path to the decoded image file
    """
    if output_path is None:
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        output_path = temp_file.name
        temp_file.close()
    
    image_data = base64.b64decode(base64_data)
    
    with open(output_path, "wb") as image_file:
        image_file.write(image_data)
        
    return output_path


def cleanup_temp_files(*file_paths: str) -> None:
    """
    Clean up temporary files.
    
    Args:
        *file_paths: Variable number of file paths to delete
    """
    for file_path in file_paths:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logging.info(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                logging.warning(f"Could not clean up file {file_path}: {e}")


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human readable format.
    
    Args:
        seconds (float): Duration in seconds
        
    Returns:
        str: Formatted duration string (e.g., "1m 30s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        float: File size in MB
    """
    if not os.path.exists(file_path):
        return 0.0
    
    return os.path.getsize(file_path) / (1024 * 1024)


def ensure_directory_exists(directory: str) -> None:
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        directory (str): Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def validate_video_file(video_path: str) -> bool:
    """
    Basic validation if a file exists and has video extension.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        bool: True if file exists with video extension, False otherwise
    """
    if not os.path.exists(video_path):
        return False
    
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
    file_extension = Path(video_path).suffix.lower()
    
    return file_extension in video_extensions


def log_generation_stats(video_path: str, generation_time: float, prompt: str = None) -> None:
    """
    Log video generation statistics.
    
    Args:
        video_path (str): Path to generated video
        generation_time (float): Time taken for generation in seconds
        prompt (str, optional): Original prompt used for generation
    """
    if os.path.exists(video_path):
        file_size_mb = get_file_size_mb(video_path)
        logging.info(f"Video generation completed:")
        logging.info(f"  - File: {video_path}")
        logging.info(f"  - File size: {file_size_mb:.1f} MB")
        logging.info(f"  - Generation time: {format_duration(generation_time)}")
        if prompt:
            logging.info(f"  - Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    else:
        logging.error(f"Generated video file not found: {video_path}") 