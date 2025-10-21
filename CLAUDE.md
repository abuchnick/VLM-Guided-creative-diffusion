# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements **VLM-Guided Adaptive Negative Prompting for Creative Generation**, a research project that enhances creative image generation in Stable Diffusion 3.5 through a closed-loop feedback mechanism. During generation, a Vision-Language Model (VLM) monitors intermediate outputs to identify dominant elements, which are dynamically accumulated as negative prompts to steer generation toward more creative outputs.

**Paper**: https://arxiv.org/abs/2510.10715
**Project Page**: https://shelley-golan.github.io/VLM-Guided-Creative-Generation/

## Development Commands

### Environment Setup

Install core dependencies:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate safetensors pillow
pip install xformers  # Optional but recommended for memory efficiency
```

For VLM support:
```bash
pip install transformers[vision]  # ViLT models
pip install qwen-vl-utils  # Qwen-VL models
```

Or install all dependencies at once:
```bash
pip install -r requirements.txt
```

### Running Image Generation

**Standard baseline generation** (no VLM guidance):
```bash
python gen_utils/generate_sd35_image.py \
    --prompt "A photo of a creative object" \
    --output_dir ./outputs \
    --seed 42
```

**VLM-guided creative generation** (main use case):
```bash
python gen_utils/generate_sd35_image.py \
    --prompt "A photo of a creative object" \
    --question "What is the main object in this image?" \
    --oracle_id "dandelin/vilt-b32-finetuned-vqa" \
    --vqa_stop_step 28 \
    --output_dir ./outputs \
    --seed 42
```

**Debug mode** with logging and intermediate predictions:
```bash
python gen_utils/generate_sd35_image.py \
    --prompt "A photo of a creative pet" \
    --question "What type of pet is it?" \
    --oracle_id "dandelin/vilt-b32-finetuned-vqa" \
    --log_negatives \
    --save_intermediate \
    --seed 42
```

## Architecture Overview

### Core Components

1. **CustomStableDiffusion3Pipeline** ([custom_model/custom_sd35.py](custom_model/custom_sd35.py))
   - Extends Hugging Face's `StableDiffusion3Pipeline` with VLM-guided adaptive negative prompting
   - Implements the closed-loop feedback mechanism that queries VLMs during denoising
   - Key additions:
     - `_latents_to_rgb()`: Converts latent predictions to RGB images for VLM analysis
     - `_flow_to_x0()`: Converts flow predictions to x0 estimates (clean image predictions) for SD3's flow matching formulation
     - Dynamic negative prompt accumulation logic integrated into the denoising loop (steps 595-665)
   - VLM guidance parameters: `question`, `oracle`, `top_k`, `freq`, `vqa_start_timestep`, `vqa_stop_timestep`

2. **Generation Script** ([gen_utils/generate_sd35_image.py](gen_utils/generate_sd35_image.py))
   - Main entry point for image generation
   - Handles CLI argument parsing and model initialization
   - Implements `QwenOracle` class for Qwen-VL model support
   - Sets up reproducibility (seeds, deterministic behavior)
   - Manages memory efficiently with CUDA cache clearing

3. **VQA Answer Cleaning** ([custom_model/custom_sd35.py](custom_model/custom_sd35.py):37-136)
   - `clean_vqa_answer()`: Normalizes VLM responses by removing artifacts ("the", "it is", "appears to be", etc.)
   - Filters overly generic responses ("yes", "no", single letters)
   - Critical for preventing noise in accumulated negative prompts

### How VLM Guidance Works

1. **Denoising loop** runs from noisiest (step 0) to cleanest (step N) timesteps
2. At each step within `[vqa_start_step, vqa_stop_step)`:
   - Compute intermediate x0 prediction using `_flow_to_x0(latents, noise_pred)`
   - Decode latents to RGB image using `_latents_to_rgb()`
   - Query VLM with user-specified questions (e.g., "What is the main object?")
   - Clean VLM responses with `clean_vqa_answer()`
   - Accumulate new detections to `detected_objects` list (avoiding duplicates)
   - Re-encode accumulated detections as negative prompt embeddings
   - Update CFG embeddings for next denoising step
3. VLM-detected features steer generation **away** from common patterns

### Supported VLM Models

- **ViLT** (`dandelin/vilt-b32-finetuned-vqa`): Fast, low VRAM, ~13s overhead for 28 steps
- **Qwen-VL** (`Qwen/Qwen2.5-VL-3B-Instruct`, `Qwen/Qwen2.5-VL-7B-Instruct`): Higher quality, slower, higher VRAM

The oracle interface expects: `oracle({"image": PIL.Image, "question": str})` → `[{"answer": str}, ...]`

### Key Implementation Details

- **Flow matching in SD3**: SD3 uses flow matching instead of DDPM noise prediction. The formula `x0 = x_t - g(x_t, t)` converts flow predictions to clean image estimates.
- **Step indexing**: Step 0 is the noisiest timestep (t≈16), step N is cleanest (t≈0). This is **opposite** to the timestep values.
- **VQA range parameters** can be specified as:
  - Absolute step indices (int): `vqa_start_step=0`, `vqa_stop_step=28`
  - Fractional (float 0.0-1.0): `vqa_start_step=0.0`, `vqa_stop_step=0.5` (halfway through)
- **Intermediate predictions** saved as `step_{i:03d}_t_{t:.4f}.png` show what the VLM analyzes at each step
- **Memory management**: Uses bfloat16, optional xformers, CUDA cache clearing, expandable segments

### Important Parameters

- `--guidance_scale` (default 4.5): CFG scale *w* from paper. Lower values = more creative diversity.
- `--vqa_stop_step` (default 28): When to stop VLM guidance. Early stopping (e.g., 14) allows refinement without creative steering in later steps.
- `--freq` (default 1): Query VLM every N steps. Higher values reduce overhead but may miss important features.
- `--topk` (default 1): Consider top K VLM answers per question. Higher values accumulate more negative prompts.
- `--clear_negatives_at_stop` (default False): Whether to clear accumulated negatives when VQA stops. Default keeps them for continued creative steering.

## Project Structure

```
VLM-Guided-creative-diffusion/
├── custom_model/
│   ├── __init__.py
│   └── custom_sd35.py           # CustomStableDiffusion3Pipeline with VLM guidance
├── gen_utils/
│   ├── __init__.py
│   └── generate_sd35_image.py   # Main generation script with CLI
├── images/                       # Paper figures
├── README.md
├── requirements.txt
└── LICENSE                       # Apache License 2.0
```

## Development Notes

- **GPU Requirements**: SD3.5-Large requires 24GB+ VRAM. Use bfloat16 and xformers for efficiency.
- **Reproducibility**: All random seeds are set (numpy, random, torch, transformers, cudnn). Use same seed for deterministic outputs.
- **Performance**: Baseline SD3.5 takes ~22s/image. VLM-guided (ViLT + 28 steps) takes ~35s/image (~13s overhead).
- **Testing changes**: Use `--log_negatives` to verify VLM detection logic and `--save_intermediate` to visualize intermediate predictions.
- **Extending VLM support**: Implement oracle class with `__call__({"image": PIL.Image, "question": str})` interface returning `[{"answer": str}]`.
