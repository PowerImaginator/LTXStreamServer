import io
import os
from pathlib import Path
from typing import Optional, List, Union, BinaryIO

import av
import numpy as np
import torch
from einops import rearrange
from PIL import Image
from transformers import (
    T5EncoderModel,
    T5Tokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)

from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.pipelines.pipeline_ltx_video import ConditioningItem, LTXVideoPipeline
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy


def create_ltx_video_pipeline(
    ckpt_path: str,
    precision: str,
    text_encoder_model_name_or_path: str,
    sampler: Optional[str] = None,
    device: Optional[str] = None,
    enhance_prompt: bool = False,
    prompt_enhancer_image_caption_model_name_or_path: Optional[str] = None,
    prompt_enhancer_llm_model_name_or_path: Optional[str] = None,
) -> LTXVideoPipeline:
    ckpt_path = Path(ckpt_path)
    assert os.path.exists(
        ckpt_path
    ), f"Ckpt path provided (--ckpt_path) {ckpt_path} does not exist"
    vae = CausalVideoAutoencoder.from_pretrained(ckpt_path)
    transformer = Transformer3DModel.from_pretrained(ckpt_path)

    # Use constructor if sampler is specified, otherwise use from_pretrained
    if sampler:
        scheduler = RectifiedFlowScheduler(
            sampler=("Uniform" if sampler.lower() == "uniform" else "LinearQuadratic")
        )
    else:
        scheduler = RectifiedFlowScheduler.from_pretrained(ckpt_path)

    text_encoder = T5EncoderModel.from_pretrained(
        text_encoder_model_name_or_path, subfolder="text_encoder"
    )
    patchifier = SymmetricPatchifier(patch_size=1)
    tokenizer = T5Tokenizer.from_pretrained(
        text_encoder_model_name_or_path, subfolder="tokenizer"
    )

    transformer = transformer.to(device)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)

    if enhance_prompt:
        prompt_enhancer_image_caption_model = AutoModelForCausalLM.from_pretrained(
            prompt_enhancer_image_caption_model_name_or_path, trust_remote_code=True
        )
        prompt_enhancer_image_caption_processor = AutoProcessor.from_pretrained(
            prompt_enhancer_image_caption_model_name_or_path, trust_remote_code=True
        )
        prompt_enhancer_llm_model = AutoModelForCausalLM.from_pretrained(
            prompt_enhancer_llm_model_name_or_path,
            torch_dtype="bfloat16",
        )
        prompt_enhancer_llm_tokenizer = AutoTokenizer.from_pretrained(
            prompt_enhancer_llm_model_name_or_path,
        )
    else:
        prompt_enhancer_image_caption_model = None
        prompt_enhancer_image_caption_processor = None
        prompt_enhancer_llm_model = None
        prompt_enhancer_llm_tokenizer = None

    vae = vae.to(torch.bfloat16)
    if precision == "bfloat16" and transformer.dtype != torch.bfloat16:
        transformer = transformer.to(torch.bfloat16)
    text_encoder = text_encoder.to(torch.bfloat16)

    # Use submodels for the pipeline
    submodel_dict = {
        "transformer": transformer,
        "patchifier": patchifier,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "scheduler": scheduler,
        "vae": vae,
        "prompt_enhancer_image_caption_model": prompt_enhancer_image_caption_model,
        "prompt_enhancer_image_caption_processor": prompt_enhancer_image_caption_processor,
        "prompt_enhancer_llm_model": prompt_enhancer_llm_model,
        "prompt_enhancer_llm_tokenizer": prompt_enhancer_llm_tokenizer,
    }

    pipeline = LTXVideoPipeline(**submodel_dict)
    pipeline = pipeline.to(device)
    return pipeline


def load_conditioning_items(
    conditioning_videos: List[Union[str, bytes, os.PathLike, BinaryIO]],
    conditioning_masks: List[List[Union[str, bytes, os.PathLike, BinaryIO]]],
    conditioning_start_frames: List[int],
    device: Optional[str] = None,
):
    conditioning_items = []

    for media_item, mask_items, start_frame in zip(
        conditioning_videos, conditioning_masks, conditioning_start_frames
    ):
        frames = []
        container = av.open(io.BytesIO(media_item))
        for frame in container.decode(video=0):
            frame_tensor = load_image_to_tensor(frame.to_image())
            frames.append(frame_tensor)
        video_tensor = torch.cat(frames, dim=2)

        mask_frames = []
        for mask_item in mask_items:
            mask_img = Image.open(io.BytesIO(mask_item)).convert("RGB")
            mask_img = np.array(mask_img).astype(np.float32) / 255.0
            mask_img = mask_img.mean(axis=2, keepdims=True)
            mask_frames.append(mask_img)
        mask_tensor = torch.Tensor(np.array(mask_frames)).to(
            device=device, dtype=torch.bfloat16
        )
        mask_tensor = rearrange(mask_tensor, "f h w c -> 1 c f h w")
        conditioning_items.append(
            ConditioningItem(video_tensor, start_frame, mask_tensor)
        )

    return conditioning_items


def load_image_to_tensor(
    image_input: Union[str, Image.Image],
) -> torch.Tensor:
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input
    else:
        raise ValueError("image_input must be either a file path or a PIL Image object")

    frame_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float()
    frame_tensor = (frame_tensor / 127.5) - 1.0
    # Create 5D tensor: (batch_size=1, channels=3, num_frames=1, height, width)
    return frame_tensor.unsqueeze(0).unsqueeze(2)


class LTXVideoSession:
    def __init__(
        self,
        ckpt_path: str,
        precision: str,
        text_encoder_model_name_or_path: str,
        sampler: Optional[str] = None,
        device: Optional[str] = None,
        enhance_prompt: bool = False,
        prompt_enhancer_image_caption_model_name_or_path: Optional[str] = None,
        prompt_enhancer_llm_model_name_or_path: Optional[str] = None,
    ):
        self.device = device
        self.pipeline = create_ltx_video_pipeline(
            ckpt_path=ckpt_path,
            precision=precision,
            text_encoder_model_name_or_path=text_encoder_model_name_or_path,
            sampler=sampler,
            device=device,
            enhance_prompt=enhance_prompt,
            prompt_enhancer_image_caption_model_name_or_path=prompt_enhancer_image_caption_model_name_or_path,
            prompt_enhancer_llm_model_name_or_path=prompt_enhancer_llm_model_name_or_path,
        )
        self.pipeline_args = {
            "num_inference_steps": 40,
            "num_images_per_prompt": 1,
            "guidance_scale": 3,
            "skip_layer_strategy": SkipLayerStrategy.AttentionValues,
            "skip_block_list": [19],
            "stg_scale": 1,
            "do_rescaling": True,
            "rescaling_scale": 0.7,
            "generator": None,
            "output_type": "pt",
            "callback_on_step_end": None,
            "height": 480,
            "width": 704,
            "num_frames": 25,
            "frame_rate": 25,
            "conditioning_items": None,
            "is_video": True,
            "vae_per_channel_normalize": True,
            "image_cond_noise_scale": 0.15,
            "decode_timestep": 0.025,
            "decode_noise_scale": 0.0125,
            "mixed_precision": False,
            "offload_to_cpu": False,
            "device": device,
            "enhance_prompt": enhance_prompt,
            "prompt": "",
            "prompt_attention_mask": None,
            "negative_prompt": "",
            "negative_prompt_attention_mask": None,
        }

    def set_pipeline_args(
        self,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        skip_layer_strategy: Optional[SkipLayerStrategy] = None,
        skip_block_list: Optional[List[int]] = None,
        stg_scale: Optional[float] = None,
        rescaling_scale: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        frame_rate: Optional[int] = None,
        conditioning_videos: Optional[
            List[Union[str, bytes, os.PathLike, BinaryIO]]
        ] = None,
        conditioning_masks: Optional[
            List[List[Union[str, bytes, os.PathLike, BinaryIO]]]
        ] = None,
        conditioning_start_frames: Optional[List[int]] = None,
        vae_per_channel_normalize: Optional[bool] = None,
        image_cond_noise_scale: Optional[float] = None,
        decode_timestep: Optional[float] = None,
        decode_noise_scale: Optional[float] = None,
        mixed_precision: Optional[bool] = None,
        offload_to_cpu: Optional[bool] = None,
        enhance_prompt: Optional[bool] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
    ):
        args = locals().copy()
        for arg_name in args:
            if (args[arg_name] is not None) and (arg_name in self.pipeline_args):
                self.pipeline_args[arg_name] = args[arg_name]

        if conditioning_videos is None:
            conditioning_items = None
        else:
            conditioning_items = load_conditioning_items(
                conditioning_videos=conditioning_videos,
                conditioning_masks=conditioning_masks,
                conditioning_start_frames=conditioning_start_frames,
                device=self.device,
            )

        self.pipeline_args["conditioning_items"] = conditioning_items
        self.pipeline_args["do_rescaling"] = (
            self.pipeline_args["rescaling_scale"] != 1.0
        )

    def update_prompt(self):
        self.pipeline.update_prompt(**self.pipeline_args)

    def update_conditioning(self):
        self.pipeline.update_conditioning(**self.pipeline_args)

    def generate(self):
        return self.pipeline(**self.pipeline_args).images[0]
