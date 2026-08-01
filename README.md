# 🎨 Advanced AI Text-to-Image & Image-to-Image Generation Studio

A high-performance, modular Python application and interactive web interface for generating and transforming images using **Stable Diffusion**, **Hugging Face Diffusers**, **PyTorch**, and **Gradio**.

Designed for speed, flexibility, and low VRAM footprint, this studio features DPM-Solver sampling, Automatic Mixed Precision (AMP), attention slicing, CPU offloading, and optional xFormers integration.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture & Flow](#-system-architecture--flow)
- [Project Directory Structure](#-project-directory-structure)
- [System Requirements](#-system-requirements)
- [Step-by-Step Setup & Installation](#-step-by-step-setup--installation)
  - [Automated Setup](#1-automated-setup)
  - [Manual Setup](#2-manual-setup)
- [Running the Application](#-running-the-application)
- [Configuration & Customization](#-configuration--customization)
- [Programmatic Usage (Python API)](#-programmatic-usage-python-api)
- [Troubleshooting & FAQs](#-troubleshooting--faqs)
- [License](#-license)

---

## 🎨 Overview

The **Advanced AI Image Generation Studio** provides an enterprise-ready pipeline to synthesize photorealistic or artistic imagery from text prompts, as well as modify existing images (Image-to-Image transformation). 

With a user-friendly Gradio web frontend, users can fine-tune generation parameters—such as prompt weighting, guidance scale, step counts, image dimensions, batch generation, and seeds—in real-time.

---

## ✨ Key Features

- **📝 Text-to-Image Synthesis**: Generate images from detailed prompt descriptions with negative prompt filtering.
- **🔄 Image-to-Image Transformation**: Re-style, enhance, or modify uploaded source images with adjustable transformation strength.
- **🚀 High-Performance Optimizations**:
  - **DPM-Solver Multistep Scheduler**: Delivers high-quality images in 20–50 steps.
  - **Automatic Mixed Precision (AMP)**: Uses FP16 precision on CUDA GPUs for accelerated inference and reduced memory usage.
  - **Memory Efficiency**: Integrated Attention Slicing (`enable_attention_slicing`) and Model CPU Offloading (`enable_model_cpu_offload`).
  - **xFormers Support**: Optional memory-efficient attention implementation.
- **📁 Output & Metadata Management**:
  - Automatic saving of generated images to `generated_images/` with unique timestamped filenames.
  - JSON metadata tracking for prompts and generation settings.
  - Grid generation & watermark utilities included.
- **💻 Interactive Web UI**: Custom-styled Gradio interface featuring real-time image galleries, file downloads, and configuration inspection.

---

## 🏗️ System Architecture & Flow

The application is organized around a lightweight UI layer, a generation orchestrator, a model-loading layer, and an output persistence layer.

```mermaid
flowchart LR
    subgraph UI["User Interface Layer (app.py)"]
        A[Gradio Blocks UI]
        B[generate_images()]
        C[generate_from_image()]
        A --> B
        A --> C
    end

    subgraph ORCH["Generation Orchestrator (image_generator.py)"]
        D[ImageGenerator]
        E[initialize()]
        F[generate_from_text()]
        G[generate_from_image()]
        B --> D
        C --> D
        D --> E
        D --> F
        D --> G
    end

    subgraph MODEL["Configuration & Model Layer"]
        H[config.py]
        I[ModelLoader]
        J[Stable Diffusion Pipeline]
        D --> H
        D --> I
        I --> J
    end

    subgraph OUTPUT["Execution & Output"]
        K[PyTorch / Diffusers Runtime]
        L[generated_images/]
        M[Gradio Gallery & File Outputs]
        J --> K
        F --> L
        G --> L
        L --> M
        K --> M
    end
```

### Detailed Execution Flow

1. **User Interaction**: The user submits prompts or images through the Gradio web dashboard in `app.py`.
2. **Request Handling**: The `generate_images()` and `generate_from_image()` handlers forward the inputs to `ImageGenerator`.
3. **Configuration & Model Setup**: `ImageGenerator` reads defaults from `config.py` and delegates model initialization to `ModelLoader`.
4. **Pipeline Execution**: `ModelLoader` loads the text-to-image or image-to-image pipeline, applies scheduler and memory optimizations, and runs inference with PyTorch and Diffusers.
5. **Output & Display**: Generated images are saved in the `generated_images/` directory and returned to the Gradio interface for display and download.

---

## 📁 Project Directory Structure

```
image_gen/
├── app.py                 # Gradio Web Application & UI Layout
├── config.py              # Central Configuration & Model Hyperparameters
├── image_generator.py     # Core Image Generation Logic (T2I & I2I)
├── model_loader.py        # Model Loading, Optimization & Pipeline Setup
├── utils.py               # Utilities (Watermarking, Grids, Metadata saving)
├── requirements.txt       # Python Dependency Specifications
├── install.ps1            # Automated PowerShell Setup Script (Windows)
├── install.sh             # Automated Bash Setup Script (Linux / macOS / Git Bash)
└── generated_images/      # Output Directory for Generated Images & Metadata
```

---

## ⚙️ System Requirements

- **Operating System**: Windows 10/11, Linux (Ubuntu 20.04+ recommended), or macOS
- **Python**: **3.14** (Required by pinned project dependencies)
- **GPU (Recommended)**: NVIDIA GPU with CUDA support and 4GB+ VRAM
- **CPU Mode**: Supported (Inference will be slower)

---

## 🚀 Step-by-Step Setup & Installation

### 1. Automated Setup

#### **Windows (PowerShell)**
Open PowerShell as Administrator in the project folder and run:
```powershell
.\install.ps1
```

#### **Linux / macOS / Git Bash**
Open a terminal in the project folder and run:
```bash
chmod +x install.sh
./install.sh
```

---

### 2. Manual Setup

If you prefer to configure the environment step-by-step:

#### **Step 1: Verify Python Version**
Ensure Python 3.14 is installed and accessible on your environment PATH:
```bash
python --version
```

#### **Step 2: Create a Virtual Environment**
```bash
python -m venv venv
```

#### **Step 3: Activate Virtual Environment**
- **Windows (PowerShell)**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **Windows (Command Prompt)**:
  ```cmd
  venv\Scripts\activate.bat
  ```
- **Linux / macOS / Git Bash**:
  ```bash
  source venv/bin/activate
  ```

#### **Step 4: Upgrade Pip & Install Dependencies**
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 🎯 Running the Application

1. Ensure your virtual environment is activated.
2. Start the web application:
   ```bash
   python app.py
   ```
3. Open your browser and navigate to:
   ```
   http://localhost:7860
   ```
   *(Or `http://127.0.0.1:7860`)*

---

## ⚙️ Configuration & Customization

All default parameters can be modified in `config.py`:

```python
@dataclass
class ModelConfig:
    # Model selection
    model_name: str = "runwayml/stable-diffusion-v1-5"
    # Alternative models: 
    # - "stabilityai/stable-diffusion-2-1"
    # - "dreamlike-art/dreamlike-photoreal-2.0"
    
    # Generation parameters
    image_size: Tuple[int, int] = (512, 512)
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    negative_prompt: str = "blurry, bad quality, distorted, ugly"
    
    # Performance & Memory Settings
    use_amp: bool = True          # Automatic Mixed Precision
    use_xformers: bool = True     # Memory efficient attention
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_batch_size: int = 4
```

---

## 🐍 Programmatic Usage (Python API)

You can import `ImageGenerator` directly in custom Python scripts:

```python
from image_generator import ImageGenerator

# Initialize generator
generator = ImageGenerator()
generator.initialize()

# Text-to-Image Generation
images, file_paths = generator.generate_from_text(
    prompt="A futuristic cyberpunk city under heavy rain, neon lights, 8k resolution",
    negative_prompt="blurry, low quality, dark",
    num_inference_steps=40,
    guidance_scale=8.0,
    width=512,
    height=512,
    num_images=2,
    seed=42
)

print(f"Images saved to: {file_paths}")

# Image-to-Image Transformation
transformed_img, path = generator.generate_from_image(
    prompt="Transform into an oil painting by Van Gogh",
    init_image="input_sample.jpg",
    strength=0.75
)

print(f"Transformed image saved to: {path}")
```

---

## ❓ Troubleshooting & FAQs

- **Out of Memory (OOM) Errors on GPU**:
  - Decrease image resolution (e.g., from `768x768` to `512x512`).
  - Generate 1 image at a time (`num_images=1`).
  - Model CPU offloading and attention slicing are enabled by default in `model_loader.py` to prevent OOM errors.
- **Python Version Mismatch**:
  - If `install.ps1` or `install.sh` fails, verify that `python` points to Python 3.14.
- **xFormers Warnings**:
  - If xFormers is not installed for your specific PyTorch/CUDA version, the application will issue a warning and gracefully fall back to standard PyTorch attention.

---

## 📜 License

This project is open-source and intended for educational and research purposes. Model weights downloaded via Hugging Face Diffusers are subject to the original model license (CreativeML Open RAIL-M for Stable Diffusion v1.5).
