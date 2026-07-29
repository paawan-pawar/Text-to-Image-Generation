# app.py
import gradio as gr
from image_generator import ImageGenerator
from config import config
import os
from PIL import Image
import torch

# Initialize generator
generator = ImageGenerator()

def generate_images(
    prompt,
    negative_prompt,
    num_steps,
    guidance_scale,
    width,
    height,
    num_images,
    seed
):
    """Generate images from text"""
    try:
        images, paths = generator.generate_from_text(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_inference_steps=int(num_steps),
            guidance_scale=float(guidance_scale),
            width=int(width),
            height=int(height),
            num_images=int(num_images),
            seed=int(seed) if seed and seed != -1 else None
        )
        
        # Add metadata to images
        metadata = f"""
        Generated Images
        Prompt: {prompt}
        Negative Prompt: {negative_prompt}
        Steps: {num_steps}
        Guidance Scale: {guidance_scale}
        Resolution: {width}x{height}
        """
        
        return images, paths, metadata
    except Exception as e:
        return None, [], f"Error: {str(e)}"

def generate_from_image(
    prompt,
    init_image,
    strength,
    negative_prompt,
    num_steps,
    guidance_scale
):
    """Generate image from existing image"""
    try:
        if init_image is None:
            return None, "Please upload an image", ""
        
        image, path = generator.generate_from_image(
            prompt=prompt,
            init_image=init_image,
            strength=float(strength),
            negative_prompt=negative_prompt if negative_prompt else None,
            num_inference_steps=int(num_steps),
            guidance_scale=float(guidance_scale)
        )
        
        metadata = f"""
        Image-to-Image Generation
        Prompt: {prompt}
        Strength: {strength}
        Steps: {num_steps}
        Guidance Scale: {guidance_scale}
        """
        
        return image, path, metadata
    except Exception as e:
        return None, f"Error: {str(e)}", ""

# Create UI
def create_interface():
    with gr.Blocks(
        title="Text-to-Image Generator"
    ) as demo:
        
        gr.Markdown(
            """
            # 🎨 Advanced Text-to-Image Generation
            ### Generate stunning images from text descriptions using Stable Diffusion
            """
        )
        
        with gr.Tabs():
            # Tab 1: Text to Image
            with gr.TabItem("📝 Text to Image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        prompt_input = gr.Textbox(
                            label="🎯 Prompt",
                            placeholder="Enter your image description here...",
                            lines=3,
                            value="A beautiful landscape with mountains, lake, and sunset, highly detailed, 4k"
                        )
                        
                        negative_prompt_input = gr.Textbox(
                            label="🚫 Negative Prompt",
                            placeholder="What to avoid in the image...",
                            lines=2,
                            value="blurry, bad quality, distorted, ugly, low resolution"
                        )
                        
                        with gr.Row():
                            num_steps = gr.Slider(
                                label="🔢 Steps",
                                minimum=10,
                                maximum=100,
                                value=50,
                                step=1
                            )
                            guidance_scale = gr.Slider(
                                label="📊 Guidance Scale",
                                minimum=1,
                                maximum=20,
                                value=7.5,
                                step=0.5
                            )
                        
                        with gr.Row():
                            width = gr.Number(
                                label="📐 Width",
                                value=512,
                                precision=0
                            )
                            height = gr.Number(
                                label="📐 Height",
                                value=512,
                                precision=0
                            )
                        
                        with gr.Row():
                            num_images = gr.Slider(
                                label="🖼️ Number of Images",
                                minimum=1,
                                maximum=4,
                                value=1,
                                step=1
                            )
                            seed = gr.Number(
                                label="🌱 Seed (-1 for random)",
                                value=-1,
                                precision=0
                            )
                        
                        generate_btn = gr.Button(
                            "🚀 Generate Images",
                            variant="primary",
                            elem_classes="generate-btn"
                        )
                    
                    with gr.Column(scale=1):
                        output_images = gr.Gallery(
                            label="Generated Images",
                            show_label=True,
                            elem_id="gallery",
                            columns=2,
                            rows=2,
                            height=600
                        )
                        output_paths = gr.File(
                            label="📁 Saved Images",
                            file_count="multiple"
                        )
                        output_metadata = gr.Textbox(
                            label="📋 Metadata",
                            lines=5
                        )
                
                # Connect the generate button
                generate_btn.click(
                    fn=generate_images,
                    inputs=[
                        prompt_input,
                        negative_prompt_input,
                        num_steps,
                        guidance_scale,
                        width,
                        height,
                        num_images,
                        seed
                    ],
                    outputs=[output_images, output_paths, output_metadata]
                )
            
            # Tab 2: Image to Image
            with gr.TabItem("🔄 Image to Image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        img2img_prompt = gr.Textbox(
                            label="🎯 Prompt",
                            placeholder="Describe the transformation...",
                            lines=3,
                            value="Turn this into a painting in the style of Van Gogh"
                        )
                        
                        img2img_input = gr.Image(
                            label="📸 Upload Image",
                            type="pil"
                        )
                        
                        with gr.Row():
                            img2img_strength = gr.Slider(
                                label="💪 Strength",
                                minimum=0.1,
                                maximum=1.0,
                                value=0.75,
                                step=0.05,
                                info="Higher values = more transformation"
                            )
                            img2img_steps = gr.Slider(
                                label="🔢 Steps",
                                minimum=10,
                                maximum=100,
                                value=50,
                                step=1
                            )
                        
                        img2img_guidance = gr.Slider(
                            label="📊 Guidance Scale",
                            minimum=1,
                            maximum=20,
                            value=7.5,
                            step=0.5
                        )
                        
                        img2img_negative = gr.Textbox(
                            label="🚫 Negative Prompt",
                            placeholder="What to avoid...",
                            lines=2,
                            value="blurry, bad quality, distorted"
                        )
                        
                        img2img_btn = gr.Button(
                            "🔄 Transform Image",
                            variant="primary",
                            elem_classes="generate-btn"
                        )
                    
                    with gr.Column(scale=1):
                        img2img_output = gr.Image(
                            label="🖼️ Transformed Image",
                            type="pil"
                        )
                        img2img_path = gr.File(label="📁 Saved Image")
                        img2img_metadata = gr.Textbox(
                            label="📋 Metadata",
                            lines=5
                        )
                
                # Connect the image-to-image button
                img2img_btn.click(
                    fn=generate_from_image,
                    inputs=[
                        img2img_prompt,
                        img2img_input,
                        img2img_strength,
                        img2img_negative,
                        img2img_steps,
                        img2img_guidance
                    ],
                    outputs=[img2img_output, img2img_path, img2img_metadata]
                )
            
            # Tab 3: Advanced Settings
            with gr.TabItem("⚙️ Advanced Settings"):
                gr.Markdown("""
                ### Advanced Configuration
                Current settings from `config.py`
                """)
                
                gr.JSON(
                    value={
                        "Model": config.model_name,
                        "Device": config.device,
                        "Default Steps": config.num_inference_steps,
                        "Default Guidance Scale": config.guidance_scale,
                        "Default Resolution": config.image_size,
                        "Use XFormers": config.use_xformers,
                        "Use AMP": config.use_amp,
                        "Max Batch Size": config.max_batch_size
                    },
                    label="Current Configuration"
                )
                
                gr.Markdown("""
                ### Tips for Better Results
                1. **Be specific** in your prompt - include style, composition, lighting
                2. **Use negative prompts** to exclude unwanted elements
                3. **Higher steps** (50-75) give better quality but slower
                4. **Guidance scale** 7-12 works best for most cases
                5. **Try different seeds** for variety in results
                6. **Use image-to-image** for style transfer or refinement
                """)
        
        # Footer
        gr.Markdown(
            """
            ---
            Built with ❤️ using Stable Diffusion and Gradio
            """
        )
    
    return demo

if __name__ == "__main__":
    # Launch the interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
            margin: auto !important;
        }
        .generate-btn {
            background: linear-gradient(45deg, #6B5B7B, #8B6B9B) !important;
            border: none !important;
            color: white !important;
        }
        .generate-btn:hover {
            transform: scale(1.02) !important;
            transition: 0.2s !important;
        }
        """,
        debug=True
    )