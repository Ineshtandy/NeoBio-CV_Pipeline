"""
I/O and file utilities.
"""

import os
from typing import List


def list_images(input_dir: str) -> List[str]:
    """
    List all image files in a directory.
    
    Supported formats: .jpg, .jpeg, .png, .bmp, .tiff
    
    Args:
        input_dir: Path to directory
        
    Returns:
        List of absolute paths to image files
    """
    if not os.path.isdir(input_dir):
        return []
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(image_extensions)
    ]
    
    return [os.path.join(input_dir, f) for f in sorted(image_files)]


def ensure_dir(path: str) -> None:
    """
    Create directory if it doesn't exist (recursively).
    
    Args:
        path: Directory path
    """
    os.makedirs(path, exist_ok=True)


def default_out_path(
    input_path: str,
    out_dir: str,
    suffix: str = "_debug.png",
) -> str:
    """
    Generate default output path based on input file.
    
    Args:
        input_path: Path to input image
        out_dir: Output directory
        suffix: Suffix to append before extension
        
    Returns:
        Full output path
    """
    ensure_dir(out_dir)
    base_name = os.path.basename(input_path)
    name, ext = os.path.splitext(base_name)
    if not ext:
        ext = ".png"
    return os.path.join(out_dir, f"{name}{suffix}")
