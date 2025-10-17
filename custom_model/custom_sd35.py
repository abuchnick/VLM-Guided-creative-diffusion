# Copyright 2025 Shelly Golan, Yotam Nitzan, Zongze Wu, Or Patashnik
# Copyright 2024 Stability AI, The HuggingFace Team and The InstantX Team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""
VLM-Guided Adaptive Negative Prompting for Stable Diffusion 3.

This module implements a custom SD3 pipeline that enhances creative image generation
through a closed-loop feedback mechanism. During the denoising process, a Vision-Language
Model (VLM) analyzes intermediate predictions to identify dominant visual features, which
are then dynamically accumulated as negative prompts to steer generation away from common
patterns.

Key Components:
    - CustomStableDiffusion3Pipeline: Main pipeline with VLM-guided adaptive negative prompting
    - clean_vqa_answer: Utility function to clean and normalize VLM responses
    
Reference:
    Paper: "VLM-Guided Adaptive Negative-Prompting for Creative Image Generation"
"""

import torch
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers import StableDiffusion3Pipeline
import os
import csv

from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
    StableDiffusion3PipelineOutput,
    PipelineImageInput,
    calculate_shift
)

def clean_vqa_answer(answer: str) -> str:
    """
    Clean and normalize VQA model responses for use as negative prompts.
    
    This function removes common VQA artifacts like "the", "it is", "appears to be",
    and other linguistic noise that would be unhelpful in negative prompts. It also
    filters out overly generic or meaningless responses.
    
    Args:
        answer: Raw answer string from VLM/VQA model
        
    Returns:
        Cleaned answer string suitable for negative prompting, or empty string if
        the answer is deemed too generic or meaningless
        
    Example:
        >>> clean_vqa_answer("It appears to be a fluffy cat")
        "fluffy cat"
        >>> clean_vqa_answer("the dog")
        "dog"
        >>> clean_vqa_answer("yes")
        ""
    """
    if not answer or not isinstance(answer, str):
        return ""
    
    # Convert to lowercase for processing
    cleaned = answer.strip().lower()
    
    # Remove leading punctuation
    while cleaned and cleaned[0] in ',-.:;!?()[]{}"\' ':
        cleaned = cleaned[1:].strip()
    
    # Remove trailing punctuation  
    while cleaned and cleaned[-1] in ',-.:;!?()[]{}"\' ':
        cleaned = cleaned[:-1].strip()
    
    # Remove common VQA artifacts and prefixes
    prefixes_to_remove = [
        'the ', 'a ', 'an ', 'it is ', 'this is ', 'that is ',
        'it\'s ', 'there is ', 'there are ', 'i see ', 'i can see ',
        'it appears to be ', 'it looks like ', 'seems to be ',
        'appears to be ', 'looks like ', 'yes, ', 'no, '
    ]
    
    for prefix in prefixes_to_remove:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break  # Only remove one prefix to avoid over-cleaning
    
    # Clean up articles and common words throughout the text
    # Split on common separators and clean each part
    separators = [',', ';', ' and ', ' or ', ' & ']
    parts = [cleaned]
    
    for sep in separators:
        new_parts = []
        for part in parts:
            new_parts.extend(part.split(sep))
        parts = new_parts
    
    # Clean each part individually
    cleaned_parts = []
    articles_to_remove = {'a', 'an', 'the'}
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        # Split into words and remove articles
        words = part.split()
        cleaned_words = []
        
        for word in words:
            word = word.strip(',-.:;!?()[]{}"\' ')
            if word and word not in articles_to_remove:
                cleaned_words.append(word)
        
        if cleaned_words:
            cleaned_part = ' '.join(cleaned_words)
            if cleaned_part and len(cleaned_part) > 1:  # Keep parts with more than 1 character
                cleaned_parts.append(cleaned_part)
    
    # Rejoin the cleaned parts
    if cleaned_parts:
        cleaned = ', '.join(cleaned_parts)
    else:
        cleaned = ""
    
    # Skip single letters/characters that aren't meaningful
    if len(cleaned) == 1 and cleaned.isalpha():
        return ""
    
    # Skip very short non-descriptive words
    skip_words = {'i', 'a', 'an', 'the', 'is', 'are', 'am', 'be', 'it', 'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'no', 'yes', '0', '1', '2', 'neither', 'no dog', 'no cat', 'none', 'no animal'}
    if cleaned in skip_words:
        return ""
    
    return cleaned


def _flow_to_x0(x_t: torch.FloatTensor, g_xt_t: torch.FloatTensor) -> torch.FloatTensor:
    """
    Convert flow prediction to x0 estimate for Stable Diffusion 3.
    
    Args:
        x_t: The noisy latent at timestep t
        g_xt_t: The model's prediction (flow/velocity)
        
    Returns:
        x0 estimate (clean image prediction)
        
    Note:
        For SD3's flow matching formulation: x0 = x_t - g(x_t, t)
    """
    return x_t - g_xt_t


class CustomStableDiffusion3Pipeline(StableDiffusion3Pipeline):
    """
    VLM-Guided Adaptive Negative Prompting Pipeline for Stable Diffusion 3.
    
    This pipeline extends the standard SD3 pipeline with a closed-loop feedback mechanism
    that uses a Vision-Language Model (VLM) to identify dominant visual features during
    denoising and dynamically accumulates them as negative prompts to steer generation
    away from common patterns.
    
    Key additions to standard SD3:
        - question: VLM questions to ask at each denoising step
        - oracle: VLM model to query for image analysis
        - top_k: Number of top VLM answers to consider
        - freq: Frequency of VLM querying (every N steps)
        - vqa_start_timestep/vqa_stop_timestep: Range of steps to apply VLM guidance
        - main_object: Optional object to append to detected features
        - clear_negatives_at_stop: Whether to clear accumulated negatives after VQA stops
        - log_negatives: Whether to log accumulated negatives during generation
        - save_intermediate: Whether to save intermediate x0 predictions
        - intermediate_dir: Directory for saving intermediate predictions
    """
    
    def _latents_to_rgb(self, latents: torch.FloatTensor):
        """
        Decode latent representations to RGB PIL images for VLM analysis.
        
        Args:
            latents: Latent tensor from the diffusion model
            
        Returns:
            PIL Image in RGB format
        """
        with torch.no_grad():
            # Scale and shift latents according to VAE configuration
            scaled_latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            # Decode using VAE
            images = self.vae.decode(scaled_latents, return_dict=False)[0]
            # Post-process to PIL format
            images = self.image_processor.postprocess(images, output_type="pil")
            return images[0]
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        skip_guidance_layers: List[int] = None,
        skip_layer_guidance_scale: float = 2.8,
        skip_layer_guidance_stop: float = 0.2,
        skip_layer_guidance_start: float = 0.01,
        mu: Optional[float] = None,
        question: Optional[List[str]] = None,
        oracle: Optional[str] = None,
        top_k: Optional[int] = None,
        freq: Optional[int] = None,
        vqa_start_timestep: Optional[Union[int, float]] = None,
        vqa_stop_timestep: Optional[Union[int, float]] = None,
        main_object: Optional[str] = None,
        clear_negatives_at_stop: bool = False,
        log_negatives: bool = False,
        save_intermediate: bool = False,
        intermediate_dir: Optional[str] = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_3` and `text_encoder_3`. If not defined, `prompt` is
                will be used instead
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used instead
            negative_prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_3` and
                `text_encoder_3`. If not defined, `negative_prompt` is used instead
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            ip_adapter_image (`PipelineImageInput`, *optional*):
                Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`torch.Tensor`, *optional*):
                Pre-generated image embeddings for IP-Adapter. Should be a tensor of shape `(batch_size, num_images,
                emb_dim)`. It should contain the negative image embedding if `do_classifier_free_guidance` is set to
                `True`. If not provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] instead of
                a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 256): Maximum sequence length to use with the `prompt`.
            skip_guidance_layers (`List[int]`, *optional*):
                A list of integers that specify layers to skip during guidance. If not provided, all layers will be
                used for guidance. If provided, the guidance will only be applied to the layers specified in the list.
                Recommended value by StabiltyAI for Stable Diffusion 3.5 Medium is [7, 8, 9].
            skip_layer_guidance_scale (`int`, *optional*): The scale of the guidance for the layers specified in
                `skip_guidance_layers`. The guidance will be applied to the layers specified in `skip_guidance_layers`
                with a scale of `skip_layer_guidance_scale`. The guidance will be applied to the rest of the layers
                with a scale of `1`.
            skip_layer_guidance_stop (`int`, *optional*): The step at which the guidance for the layers specified in
                `skip_guidance_layers` will stop. The guidance will be applied to the layers specified in
                `skip_guidance_layers` until the fraction specified in `skip_layer_guidance_stop`. Recommended value by
                StabiltyAI for Stable Diffusion 3.5 Medium is 0.2.
            skip_layer_guidance_start (`int`, *optional*): The step at which the guidance for the layers specified in
                `skip_guidance_layers` will start. The guidance will be applied to the layers specified in
                `skip_guidance_layers` from the fraction specified in `skip_layer_guidance_start`. Recommended value by
                StabiltyAI for Stable Diffusion 3.5 Medium is 0.01.
            mu (`float`, *optional*): `mu` value used for `dynamic_shifting`.
            question (`List[str]`, *optional*): List of questions to ask the VLM at each denoising step.
                Example: ["What is the main object in this image?"]
            oracle (callable, *optional*): VLM model function that takes an image and question and returns answers.
            top_k (`int`, *optional*): Number of top answers to consider from VLM responses. Default is 1.
            freq (`int`, *optional*): Frequency of VLM querying (query every N steps). Default is 1.
            vqa_start_timestep (`float` or `int`, *optional*): Starting step for VLM guidance. 
                - If float in [0.0, 1.0]: treated as fraction of total steps (e.g., 0.0 = start)
                - If int or float > 1.0: treated as absolute step index (e.g., 0 = first step)
                - If None: defaults to 0 (start from first/noisiest step)
            vqa_stop_timestep (`float` or `int`, *optional*): Ending step for VLM guidance. 
                - If float in [0.0, 1.0]: treated as fraction of total steps (e.g., 0.5 = halfway)
                - If int or float > 1.0: treated as absolute step index (e.g., 9 = step 9)
                - If None: defaults to num_inference_steps (apply until end)
            main_object (`str`, *optional*): Optional object name to append to detected features for context.
            clear_negatives_at_stop (`bool`, *optional*, defaults to `False`): Whether to clear accumulated 
                negative prompts when VQA guidance stops. If False (default), keeps accumulated negatives to 
                maintain creative steering throughout generation. If True, returns to neutral negative prompt 
                after VQA stops, allowing normal refinement in later steps.
            log_negatives (`bool`, *optional*, defaults to `False`): Whether to log accumulated negative prompts
                during generation. Useful for debugging, reproducibility, and understanding which features are
                being detected and avoided at each step.
            save_intermediate (`bool`, *optional*, defaults to `False`): Whether to save intermediate x0 predictions
                during denoising. Useful for visualizing the generation process and understanding what the VLM
                analyzes at each step.
            intermediate_dir (`str`, *optional*): Directory path where intermediate predictions should be saved.
                Only used if save_intermediate is True.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._skip_layer_guidance_scale = skip_layer_guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        if self.do_classifier_free_guidance:
            if skip_guidance_layers is not None:
                original_prompt_embeds = prompt_embeds
                original_pooled_prompt_embeds = pooled_prompt_embeds
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
            _, _, height, width = latents.shape
            image_seq_len = (height // self.transformer.config.patch_size) * (
                width // self.transformer.config.patch_size
            )
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.16),
            )
            scheduler_kwargs["mu"] = mu
        elif mu is not None:
            scheduler_kwargs["mu"] = mu
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            **scheduler_kwargs,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 6. Prepare image embeddings
        if (ip_adapter_image is not None and self.is_ip_adapter_active) or ip_adapter_image_embeds is not None:
            ip_adapter_image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

            if self.joint_attention_kwargs is None:
                self._joint_attention_kwargs = {"ip_adapter_image_embeds": ip_adapter_image_embeds}
            else:
                self._joint_attention_kwargs.update(ip_adapter_image_embeds=ip_adapter_image_embeds)

        # Initialize list for accumulating detected objects
        detected_objects = []
        if negative_prompt:
            # Handle both string and list inputs for initial negative prompt
            if isinstance(negative_prompt, str):
                detected_objects = [negative_prompt]
            elif isinstance(negative_prompt, list):
                detected_objects = list(negative_prompt)  # Create a copy
       
        # Determine VQA active range (support both fractional and absolute steps)
        if vqa_start_timestep is not None:
            if 0 < vqa_start_timestep <= 1:  # Fractional
                vqa_start_step = int(vqa_start_timestep * num_inference_steps)
            else:  # Absolute step number
                vqa_start_step = int(vqa_start_timestep)
        else:
            vqa_start_step = 0
        
        if vqa_stop_timestep is not None:
            if 0 < vqa_stop_timestep <= 1:  # Fractional
                vqa_stop_step = int(vqa_stop_timestep * num_inference_steps)
            else:  # Absolute step number
                vqa_stop_step = int(vqa_stop_timestep)
        else:
            vqa_stop_step = num_inference_steps
        
        # Validate and clamp to valid range
        vqa_start_step = max(0, min(vqa_start_step, num_inference_steps))
        vqa_stop_step = max(0, min(vqa_stop_step, num_inference_steps))
        
        # 7. Denoising loop with VLM-guided adaptive negative prompting
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    should_skip_layers = (
                        True
                        if i > num_inference_steps * skip_layer_guidance_start
                        and i < num_inference_steps * skip_layer_guidance_stop
                        else False
                    )
                    if skip_guidance_layers is not None and should_skip_layers:
                        timestep = t.expand(latents.shape[0])
                        latent_model_input = latents
                        noise_pred_skip_layers = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=original_prompt_embeds,
                            pooled_projections=original_pooled_prompt_embeds,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False,
                            skip_layers=skip_guidance_layers,
                        )[0]
                        noise_pred = (
                            noise_pred + (noise_pred_text - noise_pred_skip_layers) * self._skip_layer_guidance_scale
                        )

                # Compute intermediate clean image estimate
                latents_dtype = latents.dtype
                
                x0_hat = _flow_to_x0(latents, noise_pred)

                scheduler_output = self.scheduler.step(noise_pred, t, latents, return_dict=True)
                latents = scheduler_output.prev_sample

                # Check if VQA guidance is active for this step
                is_vqa_active = vqa_start_step <= i < vqa_stop_step

                # VLM-guided negative prompting: query VLM and update negative prompt
                if x0_hat is not None and question is not None and is_vqa_active:
                    # Convert intermediate prediction to RGB image for VLM analysis
                    x0_hat_image = self._latents_to_rgb(x0_hat)
                    
                    # Save intermediate prediction if requested
                    if save_intermediate and intermediate_dir is not None:
                        import os
                        intermediate_filename = f"step_{i:03d}_t_{t.item():.4f}.png"
                        intermediate_path = os.path.join(intermediate_dir, intermediate_filename)
                        x0_hat_image.save(intermediate_path)
                        if log_negatives:
                            print(f"  [Step {i}] Saved intermediate to: {intermediate_filename}")

                    # Query VLM with intermediate prediction
                    for j in range(top_k):
                        for k in range(len(question)):
                            answer = oracle({"image": x0_hat_image, "question": question[k]})
                            top_answer = answer[j]['answer']
                            top_answer = clean_vqa_answer(top_answer)
                            
                            # Optionally append main object for context
                            if main_object is not None:
                                top_answer = top_answer + " " + main_object

                            # Accumulate detected features to negative prompt
                            if (top_answer and                           # Not empty
                                top_answer.strip() and                   # Not just whitespace
                                top_answer not in detected_objects and   # Not already detected
                                i % freq == 0):                          # Frequency check
                                detected_objects.append(top_answer)
                                if log_negatives:
                                    print(f"  [Step {i}] Added to negatives: '{top_answer}'")

                    # Update negative prompt embeddings with accumulated detections
                    if i < len(timesteps) - 1 and self.do_classifier_free_guidance and detected_objects:
                        # Create comma-separated list from all detected objects
                        new_negative_prompt = ", ".join(detected_objects)
                        
                        if log_negatives:
                            print(f"  [Step {i}] Accumulated negatives: {new_negative_prompt}")

                        with torch.no_grad():
                            # Encode updated negative prompt with accumulated features
                            (
                                _,
                                new_negative_prompt_embeds,
                                _,
                                new_negative_pooled_prompt_embeds,
                            ) = self.encode_prompt(
                                prompt="",
                                prompt_2="",
                                prompt_3="",
                                device=device,
                                num_images_per_prompt=num_images_per_prompt,
                                do_classifier_free_guidance=True,
                                negative_prompt=new_negative_prompt,
                                negative_prompt_2=new_negative_prompt,
                                negative_prompt_3=new_negative_prompt,
                            )

                        # Update embeddings for classifier-free guidance
                        positive_prompt_embeds = prompt_embeds[batch_size * num_images_per_prompt:]
                        prompt_embeds = torch.cat([new_negative_prompt_embeds, positive_prompt_embeds], dim=0)

                        positive_pooled = pooled_prompt_embeds[batch_size * num_images_per_prompt:]
                        pooled_prompt_embeds = torch.cat([new_negative_pooled_prompt_embeds, positive_pooled], dim=0)
                
                elif question is not None and not is_vqa_active:
                    # Optionally clear negative prompt when VQA guidance becomes inactive
                    if i == vqa_stop_step and self.do_classifier_free_guidance and clear_negatives_at_stop:
                        if log_negatives:
                            print(f"  [Step {i}] VQA stopped - clearing accumulated negatives")
                        with torch.no_grad():
                            (
                                _,
                                cleared_negative_prompt_embeds,
                                _,
                                cleared_negative_pooled_prompt_embeds,
                            ) = self.encode_prompt(
                                prompt="",
                                prompt_2="",
                                prompt_3="",
                                device=device,
                                num_images_per_prompt=num_images_per_prompt,
                                do_classifier_free_guidance=True,
                                negative_prompt="",
                                negative_prompt_2="",
                                negative_prompt_3="",
                            )

                        positive_prompt_embeds = prompt_embeds[batch_size * num_images_per_prompt:]
                        prompt_embeds = torch.cat([cleared_negative_prompt_embeds, positive_prompt_embeds], dim=0)

                        positive_pooled = pooled_prompt_embeds[batch_size * num_images_per_prompt:]
                        pooled_prompt_embeds = torch.cat([cleared_negative_pooled_prompt_embeds, positive_pooled], dim=0)


                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)
