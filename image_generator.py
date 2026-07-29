# image_generator.py
import torch
from PIL import Image
import numpy as np
from typing import Optional, List, Union
from config import config
from model_loader import ModelLoader
import logging
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageGenerator:
    def __init__(self):
        self.model_loader = ModelLoader()
        self.pipe = None
        self.img2img_pipe = None
        self.output_dir = "generated_images"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def initialize(self):
        """Initialize the model"""
        if self.pipe is None:
            self.pipe = self.model_loader.load_model()
        return self
    
    def generate_from_text(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_images: int = 1,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        Generate images from text description
        
        Args:
            prompt: Text description
            negative_prompt: What to avoid
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow prompt
            width: Image width
            height: Image height
            num_images: Number of images to generate
            seed: Random seed for reproducibility
            
        Returns:
            List of PIL Images
        """
        if self.pipe is None:
            self.initialize()
        
        # Set parameters
        negative_prompt = negative_prompt or config.negative_prompt
        num_inference_steps = num_inference_steps or config.num_inference_steps
        guidance_scale = guidance_scale or config.guidance_scale
        width = width or config.image_size[0]
        height = height or config.image_size[1]
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        logger.info(f"Generating image for: {prompt}")
        
        try:
            with torch.autocast("cuda") if config.use_amp else torch.no_grad():
                result = self.pipe(
                    prompt=[prompt] * num_images,
                    negative_prompt=[negative_prompt] * num_images,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                )
            
            images = result.images
            
            # Save images
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_paths = []
            for i, img in enumerate(images):
                filename = f"gen_{timestamp}_{i}.png"
                filepath = os.path.join(self.output_dir, filename)
                img.save(filepath)
                saved_paths.append(filepath)
            
            logger.info(f"Generated {len(images)} images")
            return images, saved_paths
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def generate_from_image(
        self,
        prompt: str,
        init_image: Union[str, Image.Image],
        strength: float = 0.8,
        negative_prompt: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
    ) -> Image.Image:
        """
        Generate image from existing image (img2img)
        
        Args:
            prompt: Text description
            init_image: Initial image or path
            strength: How much to transform (0-1)
            negative_prompt: What to avoid
            
        Returns:
            PIL Image
        """
        if self.img2img_pipe is None:
            self.img2img_pipe = self.model_loader.load_img2img()
        
        # Load image if path provided
        if isinstance(init_image, str):
            init_image = Image.open(init_image).convert("RGB")
        
        # Resize image to model dimensions
        init_image = init_image.resize(config.image_size)
        
        negative_prompt = negative_prompt or config.negative_prompt
        num_inference_steps = num_inference_steps or config.num_inference_steps
        guidance_scale = guidance_scale or config.guidance_scale
        
        logger.info(f"Generating image from image with prompt: {prompt}")
        
        try:
            with torch.autocast("cuda") if config.use_amp else torch.no_grad():
                result = self.img2img_pipe(
                    prompt=prompt,
                    image=init_image,
                    strength=strength,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                )
            
            image = result.images[0]
            
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.output_dir, f"img2img_{timestamp}.png")
            image.save(filepath)
            
            return image, filepath
            
        except Exception as e:
            logger.error(f"Image-to-image generation failed: {e}")
            raise