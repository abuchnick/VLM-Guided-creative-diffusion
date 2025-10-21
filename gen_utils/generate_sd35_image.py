#!/usr/bin/env python3
"""
VLM-Guided Adaptive Negative Prompting for Stable Diffusion 3.5

This script generates images using Stable Diffusion 3.5 with optional VLM-guided
adaptive negative prompting for enhanced creative generation. The method monitors
intermediate denoiser outputs using a Vision-Language Model (VLM) to identify and
avoid dominant visual patterns during generation.

Reference:
    Paper: "VLM-Guided Adaptive Negative-Prompting for Creative Image Generation"
    
Usage:
    # Standard generation
    python generate_sd35_image.py --prompt "A photo of a creative object"
    
    # With VLM-guided negative prompting
    python generate_sd35_image.py --prompt "A photo of a creative object" \\
        --question "What is the main object in this image?" \\
        --oracle_id "dandelin/vilt-b32-finetuned-vqa"
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime
import torch
from custom_model.custom_sd35 import CustomStableDiffusion3Pipeline
from diffusers import StableDiffusion3Pipeline
from transformers import pipeline, set_seed as transformers_set_seed
import random
import numpy as np


class QwenOracle:
    """
    Qwen-VL based oracle for visual question answering.
    
    This class wraps Qwen Vision-Language models (e.g., Qwen2.5-VL-3B or 7B) for
    use in VLM-guided image generation. It provides a unified interface compatible
    with the CustomStableDiffusion3Pipeline.
    
    Args:
        model_id: HuggingFace model identifier for Qwen-VL models
        
    Example:
        >>> oracle = QwenOracle("Qwen/Qwen2.5-VL-3B-Instruct")
        >>> result = oracle({"image": pil_image, "question": "What is this?"})
        >>> print(result[0]['answer'])
    """

    def __init__(self, model_id="Qwen/Qwen2.5-VL-3B-Instruct"):
        torch.cuda.empty_cache()
        
        self.pipe = pipeline(
            "image-text-to-text", 
            model=model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

    def __call__(self, data_dict):
        """
        Query the VLM with an image and question.
        
        Args:
            data_dict: Dictionary with keys 'image' (PIL Image) and 'question' (str)
            
        Returns:
            List of dictionaries with 'answer' key containing the VLM response
        """
        from PIL import Image
        
        image = data_dict["image"]
        question = data_dict["question"]
        
        # Validate image input
        if not isinstance(image, Image.Image):
            if hasattr(image, 'convert'):
                image = image.convert('RGB')
            else:
                raise ValueError(f"Expected PIL Image, got {type(image)}")
        
        try:
            # Construct conversation for Qwen-VL API
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"Answer with a single word. \n{question}"}
                    ]
                }
            ]
            
            response = self.pipe(images=[image], text=conversation)
            
            # Parse response structure
            if response and len(response) > 0:
                if isinstance(response[0], dict) and 'generated_text' in response[0]:
                    generated_text = response[0]['generated_text']
                    if isinstance(generated_text, list):
                        for item in generated_text:
                            if isinstance(item, dict) and item.get('role') == 'assistant':
                                answer = item.get('content', '').strip()
                                return [{"answer": answer}]
                    elif isinstance(generated_text, str):
                        return [{"answer": generated_text.strip()}]
                elif isinstance(response[0], str):
                    return [{"answer": response[0].strip()}]
            
            return [{"answer": "unknown"}]
            
        except Exception as e:
            print(f"Error in QwenOracle: {e}")
            return [{"answer": "error"}]
        
        finally:
            # Memory cleanup
            torch.cuda.empty_cache()
            import gc
            gc.collect()



def main():
    """Main function for image generation with optional VLM-guided negative prompting."""
    parser = argparse.ArgumentParser(
        description="Generate images using Stable Diffusion 3.5 with optional VLM-guided adaptive negative prompting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core generation parameters
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for image generation (positive prompt)"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="Static negative prompt (optional, combined with VLM-detected features if --question is used)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated_sd35",
        help="Directory to save generated images"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility"
    )
    
    # SD3.5 generation parameters
    parser.add_argument(
        "--steps",
        type=int,
        default=28,
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=4.5,
        help="Classifier-free guidance scale (w in paper)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Generated image height in pixels"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Generated image width in pixels"
    )
    
    # VLM-guided negative prompting parameters
    parser.add_argument(
        "--question",
        nargs="+",
        action="extend",
        type=str,
        default=None,
        help="VLM question(s) for adaptive negative prompting (e.g., 'What is the main object?')"
    )
    parser.add_argument(
        "--oracle_id",
        type=str,
        default="dandelin/vilt-b32-finetuned-vqa",
        help="HuggingFace VLM model ID (supports ViLT, Qwen-VL models)"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=1,
        help="Number of top VLM answers to consider per question"
    )
    parser.add_argument(
        "--freq",
        type=int,
        default=1,
        help="VLM query frequency (query every N denoising steps)"
    )
    parser.add_argument(
        "--vqa_start_step",
        type=int,
        default=0,
        help="VQA start step index (0 = first/noisiest step, or 0.0-1.0 for fraction of total steps)"
    )
    parser.add_argument(
        "--vqa_stop_step",
        type=int,
        default=28,
        help="VQA stop step index (e.g., 28 = stop after step 28, or 0.0-1.0 for fraction of total steps)"
    )
    parser.add_argument(
        "--main_object",
        type=str,
        default=None,
        help="Optional object name to append to detected features for context"
    )
    # TODO Following were not used in the script.
    parser.add_argument(
        "--clear_negatives_at_stop",
        action="store_true",
        help="Clear accumulated negative prompts when VQA guidance stops (default: keep them)"
    )
    parser.add_argument(
        "--log_negatives",
        action="store_true",
        help="Log accumulated negative prompts during generation for debugging and reproducibility"
    )
    parser.add_argument(
        "--save_intermediate",
        action="store_true",
        help="Save intermediate x0 predictions during denoising for visualization and debugging"
    )
    parser.add_argument(
        "--intermediate_dir",
        type=str,
        default="x0_preds",
        help="Directory name for saving intermediate predictions (default: x0_preds, created inside output_dir)"
    )
    
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create intermediate predictions directory if requested
    intermediate_path = None
    if args.save_intermediate:
        intermediate_path = os.path.join(args.output_dir, args.intermediate_dir)
        os.makedirs(intermediate_path, exist_ok=True)
        print(f"Intermediate predictions will be saved to: {intermediate_path}")

    # Set all random seeds for reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    transformers_set_seed(args.seed)
    
    print(f"Random seed set to {args.seed}")

    # Memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.cuda.empty_cache()

    # Load appropriate pipeline
    print("Loading Stable Diffusion 3.5 Large model...")
    if args.question is not None:
        # Use custom pipeline for VLM-guided generation
        pipe = CustomStableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )
    else:
        # Use standard pipeline for baseline generation
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )
    
    # Ensure text encoders use bfloat16
    for encoder_name in ['text_encoder', 'text_encoder_2', 'text_encoder_3']:
        if hasattr(pipe, encoder_name):
            encoder = getattr(pipe, encoder_name)
            if encoder is not None:
                setattr(pipe, encoder_name, encoder.to(torch.bfloat16))

    # Setup VLM oracle if using adaptive negative prompting
    oracle = None
    if args.question is not None:
        print(f"Loading VLM oracle: {args.oracle_id}")
        if args.oracle_id in ["Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-3B-Instruct"]:
            oracle = QwenOracle(args.oracle_id)
        else:
            oracle = pipeline("visual-question-answering", model=args.oracle_id)
    
    # Move pipeline to GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    print(f"Using device: {device}")

    # Enable memory efficient attention if available
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("Memory efficient attention enabled (xformers)")
    except:
        print("Using default attention (xformers not available)")

    # Generate image
    generator = torch.Generator(device=device).manual_seed(args.seed)
    
    print(f"\nGenerating image with prompt: '{args.prompt}'")
    if args.question is not None:
        print(f"VLM guidance active with question(s): {args.question}")
        print(f"VQA range: steps {args.vqa_start_step} to {args.vqa_stop_step}")
        
        image = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            generator=generator,
            question=args.question,
            oracle=oracle,
            top_k=args.topk,
            freq=args.freq,
            vqa_start_timestep=args.vqa_start_step,
            vqa_stop_timestep=args.vqa_stop_step,
            main_object=args.main_object,
            clear_negatives_at_stop=args.clear_negatives_at_stop,
            log_negatives=args.log_negatives,
            save_intermediate=args.save_intermediate,
            intermediate_dir=intermediate_path
        ).images[0]
    else:
        print("Standard generation (no VLM guidance)")
        image = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            generator=generator,
        ).images[0]

    # Save generated image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sd35_image_{timestamp}_seed{args.seed}.png"
    filepath = os.path.join(args.output_dir, filename)
    image.save(filepath)
    print(f"\nImage saved to: {filepath}")


if __name__ == "__main__":
    main()
