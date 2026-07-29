# utils.py
import os
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from datetime import datetime

def add_watermark(image, text="AI Generated"):
    """Add watermark to image"""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Add watermark at bottom right
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = img.width - text_width - 20
    y = img.height - text_height - 20
    
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 128))
    return img

def save_metadata(prompt, params, filepath):
    """Save generation metadata to JSON"""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "parameters": params,
        "model": "stable-diffusion"
    }
    
    json_path = filepath.replace('.png', '.json')
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return json_path

def load_metadata(json_path):
    """Load metadata from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)

def create_image_grid(images, cols=2):
    """Create a grid of images"""
    if not images:
        return None
    
    n = len(images)
    rows = (n + cols - 1) // cols
    
    # Get image sizes
    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    max_height = max(heights)
    
    # Create grid
    grid = Image.new('RGB', (cols * max_width, rows * max_height))
    
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        grid.paste(img, (col * max_width, row * max_height))
    
    return grid