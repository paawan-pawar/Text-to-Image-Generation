# model_loader.py
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers import StableDiffusionImg2ImgPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from config import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self):
        self.device = config.device
        self.pipe = None
        self.img2img_pipe = None
        self.model_name = config.model_name
        
    def load_model(self):
        """Load the Stable Diffusion model with optimizations"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load pipeline
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
            )
            
            # Move to device
            self.pipe = self.pipe.to(self.device)
            
            # Optimizations
            if config.use_xformers:
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    logger.info("XFormers enabled")
                except Exception as e:
                    logger.warning(f"XFormers not available: {e}")
            
            # Set scheduler for faster generation
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
            
            # Enable memory optimizations
            self.pipe.enable_attention_slicing()
            if self.device == "cuda":
                self.pipe.enable_model_cpu_offload()
            
            logger.info("Model loaded successfully!")
            return self.pipe
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def load_img2img(self):
        """Load image-to-image pipeline"""
        if self.pipe is None:
            self.load_model()
        
        self.img2img_pipe = StableDiffusionImg2ImgPipeline(
            vae=self.pipe.vae,
            text_encoder=self.pipe.text_encoder,
            tokenizer=self.pipe.tokenizer,
            unet=self.pipe.unet,
            scheduler=self.pipe.scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        ).to(self.device)
        
        return self.img2img_pipe