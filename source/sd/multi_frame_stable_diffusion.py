import math
import torch
import PIL
import numpy
import cv2
import torch.nn.functional as F
from PIL import Image
from packaging import version
from typing import List, Dict, Callable, Union, Optional, Any, Tuple
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.models.controlnet import ControlNetModel
from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    DiffusionPipeline
)
from diffusers.configuration_utils import FrozenDict
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils import deprecate, logging
from transformers import CLIPTextModel, CLIPTokenizer
from .diffuser_pipeline.lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline, preprocess_image, preprocess_mask


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class StableDiffusionImg2VideoPipeline(StableDiffusionLongPromptWeightingPipeline, StableDiffusionControlNetInpaintPipeline, DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],
        scheduler,
        safety_checker,
        feature_extractor,
        requires_safety_checker: bool = True
    ):
        DiffusionPipeline.__init__(self)

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        if isinstance(controlnet, (list, tuple)):
            controlnet = MultiControlNetModel(controlnet)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )
        self.register_to_config(
            requires_safety_checker=requires_safety_checker,
        )

    def check_inputs(
        self,
        prompt,
        images,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        controlnet_conditioning_scale=1.0,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
    ):
        if height is not None and height % 8 != 0 or width is not None and width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        # `prompt` needs more sophisticated handling when there are multiple
        # conditionings.
        if isinstance(self.controlnet, MultiControlNetModel):
            if isinstance(prompt, list):
                logger.warning(
                    f"You have {len(self.controlnet.nets)} ControlNets and you have passed {len(prompt)}"
                    " prompts. The conditionings will be fixed across the prompts."
                )

        # Check `image`
        is_compiled = hasattr(F, "scaled_dot_product_attention") and isinstance(
            self.controlnet, torch._dynamo.eval_frame.OptimizedModule
        )
        if (
            isinstance(self.controlnet, ControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, ControlNetModel)
        ):
            [self.check_image(image, prompt, prompt_embeds) for image in images]
        elif (
            isinstance(self.controlnet, MultiControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
        ):
            for image in images:
                if not isinstance(image, list):
                    raise TypeError("For multiple controlnets: `images` must be type `list` of `list`")

                # When `image` is a nested list:
                # (e.g. [[canny_image_1, pose_image_1], [canny_image_2, pose_image_2]])
                elif any(isinstance(i, list) for i in image):
                    raise ValueError("A single batch of multiple conditionings are supported at the moment.")
                elif len(image) != len(self.controlnet.nets):
                    raise ValueError(
                        f"For multiple controlnets: `image` must have the same length as the number of controlnets, but got {len(images)} images and {len(self.controlnet.nets)} ControlNets."
                    )

                for image_ in image:
                    self.check_image(image_, prompt, prompt_embeds)
        else:
            assert False

        # Check `controlnet_conditioning_scale`
        if (
            isinstance(self.controlnet, ControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, ControlNetModel)
        ):
            if not isinstance(controlnet_conditioning_scale, float):
                raise TypeError("For single controlnet: `controlnet_conditioning_scale` must be type `float`.")
        elif (
            isinstance(self.controlnet, MultiControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
        ):
            if isinstance(controlnet_conditioning_scale, list):
                if any(isinstance(i, list) for i in controlnet_conditioning_scale):
                    raise ValueError("A single batch of multiple conditionings are supported at the moment.")
            elif isinstance(controlnet_conditioning_scale, list) and len(controlnet_conditioning_scale) != len(
                self.controlnet.nets
            ):
                raise ValueError(
                    "For multiple controlnets: When `controlnet_conditioning_scale` is specified as `list`, it must have"
                    " the same length as the number of controlnets"
                )
        else:
            assert False

        if len(control_guidance_start) != len(control_guidance_end):
            raise ValueError(
                f"`control_guidance_start` has {len(control_guidance_start)} elements, but `control_guidance_end` has {len(control_guidance_end)} elements. Make sure to provide the same number of elements to each list."
            )

        if isinstance(self.controlnet, MultiControlNetModel):
            if len(control_guidance_start) != len(self.controlnet.nets):
                raise ValueError(
                    f"`control_guidance_start`: {control_guidance_start} has {len(control_guidance_start)} elements but there are {len(self.controlnet.nets)} controlnets available. Make sure to provide {len(self.controlnet.nets)}."
                )

        for start, end in zip(control_guidance_start, control_guidance_end):
            if start >= end:
                raise ValueError(
                    f"control guidance start: {start} cannot be larger or equal to control guidance end: {end}."
                )
            if start < 0.0:
                raise ValueError(f"control guidance start: {start} can't be smaller than 0.")
            if end > 1.0:
                raise ValueError(f"control guidance end: {end} can't be larger than 1.0.")

    # Copied from diffusers.pipelines.controlnet.pipeline_controlnet.StableDiffusionControlNetPipeline.prepare_image
    def prepare_control_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        images: List[PipelineImageInput] = None,
        mask_images: List[PipelineImageInput] = None,
        control_images: Union[List[PipelineImageInput], List[List[PipelineImageInput]]] = None,
        blur_radius: int = 0,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        strength: float = 0.8,
        num_images_per_prompt: Optional[int] = 1,
        add_predicted_noise: Optional[bool] = False,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_embeddings_multiples: Optional[int] = 3,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        is_cancelled_callback: Optional[Callable[[], bool]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        controlnet_conditioning_scale: Union[float, List[float]] = 0.5,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            images (`List[torch.FloatTensor]` or `List[PIL.Image.Image]`):
                List of frames. Every element, `Image`, or tensor representing an image batch, that will be used as the
                starting point for the process.
            mask_images (`List[torch.FloatTensor]` or `List[PIL.Image.Image]`):
                `Image`, or tensor representing an image batch, to mask `image`. White pixels in the mask will be
                replaced by noise and therefore repainted, while black pixels will be preserved. If `mask_image` is a
                PIL image, it will be converted to a single channel (luminance) before use. If it's a tensor, it should
                contain one color channel (L) instead of 3, so the expected shape would be `(B, H, W, 1)`.
            control_images (`List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[List[torch.FloatTensor]]`, or `List[List[PIL.Image.Image]]`):
                List of control images. Every element, `Image`, or tensor representing an image batch, that will be used as the
                control image for the process. If `control_images` is a list of lists, then every element of the outer list will
                be used as a set of control images for the corresponding element of `images`. This is for MultiControlNet.
            blur_radius (`int`, *optional*, defaults to 0):
                The radius of the blur filter applied to the mask image.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1.
                `image` will be used as a starting point, adding more noise to it the larger the `strength`. The
                number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added
                noise will be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            add_predicted_noise (`bool`, *optional*, defaults to True):
                Use predicted noise instead of random noise when constructing noisy versions of the original image in
                the reverse diffusion process
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
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
            max_embeddings_multiples (`int`, *optional*, defaults to `3`):
                The max multiple length of prompt embeddings compared to the max output length of text encoder.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            is_cancelled_callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. If the function returns
                `True`, the inference will be cancelled.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 0.5):
                The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original `unet`. If multiple ControlNets are specified in `init`, you can set
                the corresponding scale as a list.
            guess_mode (`bool`, *optional*, defaults to `False`):
                The ControlNet encoder tries to recognize the content of the input image even if you remove all
                prompts. A `guidance_scale` value between 3.0 and 5.0 is recommended.
            control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the ControlNet starts applying.
            control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the ControlNet stops applying.

        Returns:
            `None` if cancelled by `is_cancelled_callback`,
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        # 1. Align format for control guidance
        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [
                control_guidance_end
            ]

        # 2. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 3. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt=prompt,
            images=images,
            height=height,
            width=width,
            callback_steps=callback_steps,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
        )

        # 4. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions

        # 5. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            max_embeddings_multiples,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        dtype = prompt_embeds.dtype

        # 6. Preprocess image and mask
        if images is not None:
            for i, image in enumerate(images):
                if isinstance(image, PIL.Image.Image):
                    image = preprocess_image(image, batch_size)
                    image = image.to(device=self.device, dtype=dtype)
                else:
                    image = None
                images[i] = image
        # 6.1 Blur mask edges
        if blur_radius:
            mask_images = [cv2.GaussianBlur(mask_image, (blur_radius, blur_radius), sigmaX=0) for mask_image in mask_images]

        if mask_images is not None:
            masks = []
            for i, mask_image in enumerate(mask_images):
                if isinstance(mask_image, PIL.Image.Image):
                    mask = preprocess_mask(mask_image, batch_size, self.vae_scale_factor)
                    mask = mask.to(device=self.device, dtype=dtype)
                    mask = torch.cat([mask] * num_images_per_prompt)
                else:
                    mask_image = None
                masks.append(mask_image)

        # 7. Prepare control image
        if isinstance(controlnet, ControlNetModel):
            control_images: List = [self.prepare_control_image(
                image=control_image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=guess_mode,
            ) for control_image in control_images]

        elif isinstance(controlnet, MultiControlNetModel):
            all_control_images = []
            for control_image in control_images:

                control_images_ = []

                for control_image_ in control_image:
                    control_image_ = self.prepare_control_image(
                        image=control_image_,
                        width=width,
                        height=height,
                        batch_size=batch_size * num_images_per_prompt,
                        num_images_per_prompt=num_images_per_prompt,
                        device=device,
                        dtype=controlnet.dtype,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        guess_mode=guess_mode,
                    )

                    control_images_.append(control_image_)

                all_control_images.append(control_images_)

            control_images: List[List] = all_control_images
        else:
            assert False

        # 8. Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device, images is None)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 9. Prepare latent variables
        if images is not None:
            num_images = len(images)
            latents_list = []
            init_latents_orig_list = []
            noise_list = []
            for image in images:
                latents, init_latents_orig, noise = self.prepare_latents(
                    image,
                    latent_timestep,
                    num_images_per_prompt,
                    batch_size,
                    self.unet.config.in_channels,
                    height,
                    width,
                    dtype,
                    device,
                    generator,
                    latents,
                )
                latents_list.append(latents)
                init_latents_orig_list.append(init_latents_orig)
                noise_list.append(noise)
        else:
            num_images = 0
            latents, init_latents_orig, noise = self.prepare_latents(
                None,
                latent_timestep,
                num_images_per_prompt,
                batch_size,
                self.unet.config.in_channels,
                height,
                width,
                dtype,
                device,
                generator,
                latents,
            )
            latents_list = [latents]
            init_latents_orig_list = [init_latents_orig]
            noise_list = [noise]

        # 10. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 10.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < i or (i + 1) / len(timesteps) > e)
                for i, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        # 11. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for s, t in enumerate(timesteps):
                # zero value
                denoised_latents_frame_list = []
                for j in range(num_images):
                    latents_frame = latents_list[j]
                    control_image = control_images[j]

                    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                    latent_model_input = torch.cat([latents_frame] * 2) if do_classifier_free_guidance else latents_frame
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # controlnet(s) inference
                    if guess_mode and do_classifier_free_guidance:
                        # Infer ControlNet only for the conditional batch.
                        control_model_input = latents
                        control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                        controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                    else:
                        control_model_input = latent_model_input
                        controlnet_prompt_embeds = prompt_embeds

                    if isinstance(controlnet_keep[j], list):
                        cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[s])]
                    else:
                        controlnet_cond_scale = controlnet_conditioning_scale
                        if isinstance(controlnet_cond_scale, list):
                            controlnet_cond_scale = controlnet_cond_scale[0]
                        cond_scale = controlnet_cond_scale * controlnet_keep[s]

                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        control_model_input,
                        t,
                        encoder_hidden_states=controlnet_prompt_embeds,
                        controlnet_cond=control_image,
                        conditioning_scale=cond_scale,
                        guess_mode=guess_mode,
                        return_dict=False,
                    )

                    if guess_mode and do_classifier_free_guidance:
                        # Infered ControlNet only for the conditional batch.
                        # To apply the output of ControlNet to both the unconditional and conditional batches,
                        # add 0 to the unconditional batch to keep it unchanged.
                        down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                        mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample

                    del latent_model_input

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_cond, guidance_rescale=guidance_rescale)

                        del noise_pred_uncond, noise_pred_cond

                    # compute the previous noisy sample x_t -> x_t-1
                    latents_frame = self.scheduler.step(noise_pred, t, latents_frame, **extra_step_kwargs, return_dict=False)[0]
                    self.scheduler._step_index -= 1

                    # handle inpainting
                    if mask_images is not None:
                        mask = masks[j]
                        init_latents_orig = init_latents_orig[j]
                        noise = noise_list[j]
                        # masking
                        if add_predicted_noise:
                            init_latents_proper = self.scheduler.add_noise(
                                init_latents_orig, noise_pred_uncond, torch.tensor([t])
                            )
                        else:
                            init_latents_proper = self.scheduler.add_noise(init_latents_orig, noise, torch.tensor([t]))
                        latents = (init_latents_proper * mask) + (latents * (1 - mask))

                    denoised_latents_frame_list.append(latents_frame)

                self.scheduler._step_index += 1

                # Overlapping
                for i, latents_frame in enumerate(denoised_latents_frame_list):
                    latents_list[i].zero_()
                    count = 0
                    for j in range(num_images):
                        distance = abs(i - j)
                        weight = math.exp(-distance)
                        latents_list[i] += denoised_latents_frame_list[j] * weight
                        count += weight
                    latents_list[i] /= count

                # call the callback, if provided
                if s == len(timesteps) - 1 or ((s + 1) > num_warmup_steps and (s + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if s % callback_steps == 0:
                        if callback is not None:
                            callback(s, t, latents_list)
                        if is_cancelled_callback is not None and is_cancelled_callback():
                            return None

        if output_type == "latent":
            images = latents_list
            # has_nsfw_concept = None
        elif output_type == "pil":
            # 12. Post-processing
            images = [self.decode_latents(latents) for latents in latents_list]

            # 13. Run safety checker
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

            # 14. Convert to PIL
            images = [self.numpy_to_pil(image) for image in images]
        else:
            # 12. Post-processing
            images = [self.decode_latents(latents) for latents in latents_list]

            # 13. Run safety checker
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return images, None

        return StableDiffusionPipelineOutput(images=images, nsfw_content_detected=None)


def make_canny_images(images: List[Image.Image], threshold1=100, threshold2=200) -> List[Image.Image]:
    """
    Make canny images from a list of PIL images.
    """
    canny_images = []
    for image in images:
        image = numpy.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.Canny(image, threshold1, threshold2)
        image = Image.fromarray(image)
        canny_images.append(image)
    return canny_images


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


def load_pipe(model_path, control_net_model_paths, use_safetensors=True, scheduler_type="euler-ancestral") -> StableDiffusionImg2VideoPipeline:
    # load control net and stable diffusion v1-5 model into one pipeline.

    if isinstance(control_net_model_paths, str):
        controlnet = ControlNetModel.from_pretrained(
            control_net_model_paths,
            torch_dtype=torch.float16
        )
    elif isinstance(control_net_model_paths, list):
        controlnets = []
        for control_net_model_path in control_net_model_paths:
            controlnet = ControlNetModel.from_pretrained(
                control_net_model_path,
                torch_dtype=torch.float16
            )
            controlnets.append(controlnet)
        controlnet = MultiControlNetModel(controlnets)

    pipe: StableDiffusionImg2VideoPipeline = StableDiffusionImg2VideoPipeline.from_single_file(
        model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        use_safetensors=use_safetensors,
        local_files_only=True,
        scheduler_type=scheduler_type,
    ).to("cuda")

    pipe.safety_checker = None

    # Enable XFormers
    pipe.enable_xformers_memory_efficient_attention(attention_op=None)
    pipe.enable_model_cpu_offload()
    return pipe


def make_inpaint_condition(image, image_mask):
    image = numpy.array(image.convert("RGB")).astype(numpy.float32) / 255.0
    image_mask = numpy.array(image_mask.convert("L")).astype(numpy.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = numpy.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


if __name__ == "__main__":
    import random
    from pathlib import Path
    pipe: StableDiffusionImg2VideoPipeline = load_pipe(
        model_path="models/Stable-Diffusion/AIDv2.10.safetensors",
        use_safetensors=True,
        scheduler_type="euler-ancestral"
    )

    neg_emb_path, neg_emb_token = "models/embeddings/aid210.pt", 'aid210'
    pipe.load_textual_inversion(neg_emb_path, token=neg_emb_token)

    prompt = "best quality, masterpiece, by ask, 1boy, solo, male focus, handsome"
    neg_prompt = neg_emb_token

    width = 848
    height = 480

    image_dir = Path("test/multidiffusion/frames")
    images = [Image.open(impath) for impath in image_dir.iterdir()]
    images = [image.resize((width, height), resample=Image.Resampling.LANCZOS) for image in images]
    canny_images = make_canny_images(images)

    seed = random.randint(0, 9999999999)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    images = pipe.__call__(
        prompt=prompt,
        negative_prompt=neg_prompt,
        images=images,
        mask_images=None,
        control_images=canny_images,
        width=width,
        height=height,
        num_inference_steps=32,
        strength=0.75,
        generator=generator,
        guidance_scale=9
    ).images

    output_dir = Path("test/multidiffusion/outputs/frames")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, image in enumerate(images):
        image[0].save(output_dir.joinpath(f"frame_{i}.png"))
