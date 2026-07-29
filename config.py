# config.py
import torch
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class ModelConfig:
    # Model selection
    model_name: str = "runwayml/stable-diffusion-v1-5"
    # Alternative models: "stabilityai/stable-diffusion-2-1", 
    # "dreamlike-art/dreamlike-photoreal-2.0"
    
    # Generation parameters
    image_size: Tuple[int, int] = (512, 512)
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    negative_prompt: str = "blurry, bad quality, distorted, ugly"
    
    # Performance
    use_amp: bool = True  # Automatic Mixed Precision
    use_xformers: bool = True  # Memory efficient attention
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Batch settings
    max_batch_size: int = 4

config = ModelConfig()