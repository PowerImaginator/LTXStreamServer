import random
import numpy as np
import torch
from PIL import Image
from LTXVideoSession import LTXVideoSession

device = None
generator = None
session = None


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def has_pipeline():
    return session is not None


def create_pipeline():
    global device
    global generator
    global session
    device = get_device()
    generator = torch.Generator(device=device)
    session = LTXVideoSession(
        ckpt_path="checkpoints/ltxv-2b-0.9.6-distilled-04-25.safetensors",
        precision="bfloat16",
        text_encoder_model_name_or_path="PixArt-alpha/PixArt-XL-2-1024-MS",
        device=device,
        enhance_prompt=False,
        prompt_enhancer_image_caption_model_name_or_path=None,
        prompt_enhancer_llm_model_name_or_path=None,
    )


def set_pipeline_args(seed_and_pipeline_args):
    if "seed" in seed_and_pipeline_args:
        seed_everything(seed_and_pipeline_args["seed"])
        generator.manual_seed(seed_and_pipeline_args["seed"])
    session.set_pipeline_args(**seed_and_pipeline_args["pipeline_args"])


def update_prompt():
    session.update_prompt()


def update_conditioning(pop_latents=None, condition_all_non_popped_latents=False):
    session.update_conditioning(
        pop_latents=pop_latents,
        condition_all_non_popped_latents=condition_all_non_popped_latents,
    )


def generate():
    video = session.generate()
    video_np = video.permute(1, 2, 3, 0).cpu().float().numpy()
    video_np = (video_np * 255).astype(np.uint8)
    return video_np


if __name__ == "__main__":
    TOTAL_FRAMES_PER_GENERATION = 25
    POP_LATENTS_PER_GENERATION = 2

    if not has_pipeline():
        create_pipeline()

    set_pipeline_args(
        {
            "seed": 123,
            "pipeline_args": {
                "width": 1216,
                "height": 704,
                "num_frames": TOTAL_FRAMES_PER_GENERATION,
                "frame_rate": 30,
                "num_inference_steps": 8,
                "prompt": "A clear, turquoise river flows through a rocky canyon, cascading over a small waterfall and forming a pool of water at the bottom.The river is the main focus of the scene, with its clear water reflecting the surrounding trees and rocks. The canyon walls are steep and rocky, with some vegetation growing on them. The trees are mostly pine trees, with their green needles contrasting with the brown and gray rocks. The overall tone of the scene is one of peace and tranquility.",
                "negative_prompt": "",
                "image_cond_noise_scale": 0.0,
                "decode_timestep": 0.05,
                "decode_noise_scale": 0.025,
            },
        }
    )
    update_prompt()

    written_frames_counter = 0
    while True:
        is_first_generation = written_frames_counter == 0

        update_conditioning(
            pop_latents=None if is_first_generation else POP_LATENTS_PER_GENERATION,
            condition_all_non_popped_latents=True,
        )

        output_frames = generate()
        output_frames_to_keep = (
            TOTAL_FRAMES_PER_GENERATION
            if is_first_generation
            else (POP_LATENTS_PER_GENERATION * 8)
        )
        output_frames = output_frames[-output_frames_to_keep:]
        for frame in output_frames:
            Image.fromarray(frame).save("outputs/{}.png".format(written_frames_counter))
            written_frames_counter += 1
