"""
Image Processing Utilities
Image preprocessing methods consistent with baseline_generator
"""

from PIL import Image
from typing import Tuple


def resize_and_pad_image(image: Image.Image, target_size: int = 1024) -> Image.Image:
    """
    Resize image to target_size x target_size while maintaining aspect ratio and padding with white borders

    Processing method:
    1. Scale the longer side to target_size
    2. Scale the shorter side proportionally
    3. Pad with white borders to target_size x target_size

    Args:
        image: PIL image object
        target_size: Target size (default 1024)

    Returns:
        Processed PIL image (target_size x target_size)
    """
    # Get original dimensions
    width, height = image.size

    # Calculate scale factor (based on longer side)
    if width >= height:
        scale = target_size / width
        new_width = target_size
        new_height = int(height * scale)
    else:
        scale = target_size / height
        new_height = target_size
        new_width = int(width * scale)

    # Use LANCZOS resampling for scaling
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create white background
    padded_image = Image.new('RGB', (target_size, target_size), (255, 255, 255))

    # Calculate centered paste position
    paste_x = (target_size - new_width) // 2
    paste_y = (target_size - new_height) // 2

    # Paste the resized image onto the white background
    padded_image.paste(resized_image, (paste_x, paste_y))

    return padded_image


def load_and_process_image(image_path: str, target_size: int = 1024) -> Image.Image:
    """
    Load and process image

    Args:
        image_path: Image file path
        target_size: Target size (default 1024)

    Returns:
        Processed PIL image (RGB mode, target_size x target_size)
    """
    # Load image and convert to RGB mode
    image = Image.open(image_path).convert('RGB')

    # Apply resize and padding
    processed_image = resize_and_pad_image(image, target_size)

    return processed_image


def batch_process_images(image_list: list, target_size: int = 1024) -> list:
    """
    Batch process a list of loaded PIL images

    Args:
        image_list: List of PIL image objects
        target_size: Target size (default 1024)

    Returns:
        List of processed PIL images
    """
    return [resize_and_pad_image(img, target_size) for img in image_list]
