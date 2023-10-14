import torch
import PIL
import numpy
import cv2
import torch.nn.functional as F
import tqdm
from PIL import Image
from packaging import version
from typing import List, Dict, Callable, Union, Optional, Any, Tuple
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.utils import PIL_INTERPOLATION
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.models.controlnet import ControlNetModel
from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    DiffusionPipeline,
)
from diffusers.configuration_utils import FrozenDict
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils import deprecate, logging
from transformers import CLIPTextModel, CLIPTokenizer
from .lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline, preprocess_image

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def preprocess_mask(mask, batch_size, scale_factor=8, blur_radius=0):
    if not isinstance(mask, torch.FloatTensor):
        mask = mask.convert("L")
        w, h = mask.size
        w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8
        mask = mask.resize((w // scale_factor, h // scale_factor), resample=PIL_INTERPOLATION["nearest"])
        mask = numpy.array(mask).astype(numpy.float32)
        if blur_radius:  # Blur mask edges
            if blur_radius % 2 == 0:
                blur_radius += 1  # must be odd
            mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), sigmaX=0)
        mask /= 255.0
        mask = numpy.tile(mask, (4, 1, 1))
        mask = numpy.vstack([mask[None]] * batch_size)
        mask = 1 - mask  # repaint white, keep black
        mask = torch.from_numpy(mask)
        return mask

    else:
        valid_mask_channel_sizes = [1, 3]
        # if mask channel is fourth tensor dimension, permute dimensions to pytorch standard (B, C, H, W)
        if mask.shape[3] in valid_mask_channel_sizes:
            mask = mask.permute(0, 3, 1, 2)
        elif mask.shape[1] not in valid_mask_channel_sizes:
            raise ValueError(
                f"Mask channel dimension of size in {valid_mask_channel_sizes} should be second or fourth dimension,"
                f" but received mask of shape {tuple(mask.shape)}"
            )
        # (potentially) reduce mask channel dimension from 3 to 1 for broadcasting to latent shape
        mask = mask.mean(dim=1, keepdim=True)
        h, w = mask.shape[-2:]
        h, w = (x - x % 8 for x in (h, w))  # resize to integer multiple of 8
        mask = torch.nn.functional.interpolate(mask, (h // scale_factor, w // scale_factor))
        return mask


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
        control_images,
        masks,
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

        # Check images and masks
        if images is not None:
            [self.check_image(image, prompt, prompt_embeds) for image in images]
        if masks is not None:
            [self.check_image(mask, prompt, prompt_embeds) for mask in masks]

        # Images, masks and control images should the same length
        if masks is None:
            pass
        elif images is None:
            raise ValueError(
                "If `masks` are provided, `images` must be provided as well. Got `images` = None and `masks` ="
                f" {masks}."
            )
        elif len(images) != len(masks):
            raise ValueError(
                f"`images` and `masks` must have the same length but got `images` {len(images)} != `masks`"
                f" {len(masks)}."
            )

        if images is not None and control_images is not None and len(control_images) != len(images):
            raise ValueError(
                f"`control_images` and `images` must have the same length but got `control_images`"
                f" {len(control_images)} != `images` {len(images)}."
            )

        # Check control images
        is_compiled = hasattr(F, "scaled_dot_product_attention") and isinstance(
            self.controlnet, torch._dynamo.eval_frame.OptimizedModule
        )
        if control_images is None:
            pass
        elif (
            isinstance(self.controlnet, ControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, ControlNetModel)
        ):
            [self.check_image(image, prompt, prompt_embeds) for image in control_images]
        elif (
            isinstance(self.controlnet, MultiControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
        ):
            for control_images_ in control_images:
                if not isinstance(control_images_, list):
                    raise TypeError("For multiple controlnets: `images` must be type `list` of `list`")

                # When `image` is a nested list:
                # (e.g. [[canny_image_1, pose_image_1], [canny_image_2, pose_image_2]])
                elif any(isinstance(i, list) for i in control_images_):
                    raise ValueError("A single batch of multiple conditionings are supported at the moment.")
                elif len(control_images_) != len(self.controlnet.nets):
                    raise ValueError(
                        f"For multiple controlnets: `image` must have the same length as the number of controlnets, but got {len(control_images_)} images and {len(self.controlnet.nets)} ControlNets."
                    )

                for control_image_ in control_images_:
                    self.check_image(control_image_, prompt, prompt_embeds)
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

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        images: List[PipelineImageInput] = None,
        masks: List[PipelineImageInput] = None,
        control_images: Union[List[PipelineImageInput], List[List[PipelineImageInput]]] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        strength: float = 0.8,
        blur_radius: int = 4,
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
            masks (`List[torch.FloatTensor]` or `List[PIL.Image.Image]`):
                `Image`, or tensor representing an image batch, to mask `image`. White pixels in the mask will be
                replaced by noise and therefore repainted, while black pixels will be preserved. If `mask_image` is a
                PIL image, it will be converted to a single channel (luminance) before use. If it's a tensor, it should
                contain one color channel (L) instead of 3, so the expected shape would be `(B, H, W, 1)`.
            control_images (`List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[List[torch.FloatTensor]]`, or `List[List[PIL.Image.Image]]`):
                List of control images. Every element, `Image`, or tensor representing an image batch, that will be used as the
                control image for the process. If `control_images` is a list of lists, then every element of the outer list will
                be used as a set of control images for the corresponding element of `images`. This is for MultiControlNet.
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
            blur_radius (`int`, *optional*, defaults to 0):
                The radius of the blur filter applied to the mask image.
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
            masks=masks,
            control_images=control_images,
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

        is_img2img = images is not None
        is_inpainting = masks is not None
        use_controlnet = control_images is not None

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
        if is_img2img:
            for i, image in enumerate(images):
                if isinstance(image, PIL.Image.Image):
                    image = image.resize((width, height), resample=Image.Resampling.LANCZOS)
                    image = preprocess_image(image, batch_size)
                    image = image.to(device=self.device, dtype=dtype)
                else:
                    image = None
                images[i] = image

        if is_inpainting:
            for i, mask_image in enumerate(masks):
                if isinstance(mask_image, PIL.Image.Image):
                    mask_image = mask_image.resize((width, height), resample=Image.Resampling.LANCZOS)
                    mask = preprocess_mask(mask_image, batch_size, self.vae_scale_factor, blur_radius)
                    mask = mask.to(device=self.device, dtype=dtype)
                    mask = torch.cat([mask] * num_images_per_prompt)
                else:
                    mask = None
                masks[i] = mask

        # 7. Prepare control image
        if not use_controlnet:
            pass
        elif isinstance(controlnet, ControlNetModel):
            for i, control_image in enumerate(control_images):
                control_image = control_image.resize((width, height), resample=Image.Resampling.LANCZOS)
                control_image = self.prepare_control_image(
                    image=control_image,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )
                control_images[i] = control_image

        elif isinstance(controlnet, MultiControlNetModel):
            for i, control_image_list in enumerate(control_images):
                for j, control_image in enumerate(control_image_list):
                    control_image = control_image.resize((width, height), resample=Image.Resampling.LANCZOS)
                    control_image = self.prepare_control_image(
                        image=control_image,
                        width=width,
                        height=height,
                        batch_size=batch_size * num_images_per_prompt,
                        num_images_per_prompt=num_images_per_prompt,
                        device=device,
                        dtype=controlnet.dtype,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        guess_mode=guess_mode,
                    )
                    control_image_list[j] = control_image
                control_images[i] = control_image_list
        else:
            raise ValueError(f"Invalid controlnet type: {type(controlnet)}")

        # 8. Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device, images is None)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 9. Prepare latent variables
        if is_img2img or use_controlnet:
            num_images = len(images) if images is not None else 0
            if use_controlnet:
                num_images = max(num_images, len(control_images))

            latents_list = []
            init_latents_orig_list = []
            noise_list = []
            for i in range(num_images):
                image = images[i] if is_img2img and i < len(images) else None
                init_latents, init_latents_orig, noise = self.prepare_latents(
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
                latents_list.append(init_latents)
                init_latents_orig_list.append(init_latents_orig)
                noise_list.append(noise)
        else:
            num_images = 0
            init_latents, init_latents_orig, noise = self.prepare_latents(
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
            latents_list = [init_latents]
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

                if isinstance(controlnet_keep[s], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[s])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[s]

                for j in tqdm.tqdm(range(num_images), desc="Denoising", leave=False):
                    latents_frame = latents_list[j]

                    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                    latent_model_input = torch.cat([latents_frame] * 2) if do_classifier_free_guidance else latents_frame
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    if use_controlnet and j < len(control_images):
                        control_image = control_images[j]

                        if guess_mode and do_classifier_free_guidance:
                            # Infer ControlNet only for the conditional batch.
                            control_model_input = latents_frame
                            control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                            controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                        else:
                            control_model_input = latent_model_input
                            controlnet_prompt_embeds = prompt_embeds

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
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                    ).sample

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_cond, guidance_rescale=guidance_rescale)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents_frame = self.scheduler.step(noise_pred, t, latents_frame, **extra_step_kwargs, return_dict=False)[0]
                    self.scheduler._step_index -= 1

                    # handle inpainting
                    if masks is not None:
                        mask = masks[j]
                        init_latents_orig = init_latents_orig_list[j]
                        noise = noise_list[j]
                        # masking
                        if add_predicted_noise:
                            init_latents_proper = self.scheduler.add_noise(
                                init_latents_orig, noise_pred_uncond, torch.tensor([t])
                            )
                        else:
                            init_latents_proper = self.scheduler.add_noise(init_latents_orig, noise, torch.tensor([t]))
                        latents_frame = (init_latents_proper * mask) + (latents_frame * (1 - mask))

                    denoised_latents_frame_list.append(latents_frame)

                if self.scheduler._step_index is not None:
                    self.scheduler._step_index += 1

                # TODO: Do overlapping using some algorithm
                # self.overlap(latents_list, corr_map=None, generator=generator)

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

    # TODO: Complete this function
    def overlap(self, latents_list: List[torch.Tensor], corr_map: Dict, generator: torch.Generator):
        """
        Do multi-diffusion overlapping on a list of frame latents according to the corresponding map.
        :param latents_list: A list of frame latents. Each element is a tensor of shape [B, C, H, W].
        :param corr_map: A dictionary of correspondence map. Each vertex id (key) is a dictionary of the following format:
            {
                'position': [x, y],  # The pixel position of the vertex in the image
            }
        :param generator: A torch generator used to sample latent dist.
        :return: A list of overlapped frame latents.
        """
        def _encode(image):
            latent_dist = self.vae.encode(image).latent_dist
            latents = latent_dist.sample(generator=generator)
            latents = self.vae.config.scaling_factor * latents
            return latents

        def _decode(latents):
            latents = 1 / self.vae.config.scaling_factor * latents
            image = self.vae.decode(latents).sample
            return image

        images = [_decode(latents) for latents in latents_list]  # [B, C, H, W]
        value = [torch.zeros_like(img) for img in images]  # [B, C, H, W]
        count = value.copy()
        screen_h, screen_w = images[0].shape[:2]
        for id, msg in corr_map.items():  # For each vertex in the correspondence map
            for i, pos in enumerate(msg['position']):  # For each frame position of the vertex
                w, h = pos
                if w >= 0 and w < screen_w and h >= 0 and h < screen_h:
                    value[i][:, :, h, w] += images[i][:, :, h, w]
                    count[i][:, :, h, w] += 1

        for i in len(value):
            value[i] = torch.where(count[i] > 0, value[i] / count[i], value[i])
            value[i] = _encode(value[i])
        return value


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


def make_inpaint_condition(image, image_mask):
    image = numpy.array(image.convert("RGB")).astype(numpy.float32) / 255.0
    image_mask = numpy.array(image_mask.convert("L")).astype(numpy.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = numpy.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image
