import sys

# Needed to support relative imports? (e.g. LTXVideoSession)
sys.path.append(".")

import random
import numpy as np
import torch
from celery import Celery

from LTXVideoSession import LTXVideoSession

app = Celery("ltxstream-tasks", backend="rpc://", broker="pyamqp://")


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


device = None
generator = None
session = None


@app.task
def has_pipeline():
    return session is not None


@app.task
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


@app.task
def set_pipeline_args(seed_and_pipeline_args):
    if "seed" in seed_and_pipeline_args:
        seed_everything(seed_and_pipeline_args["seed"])
        generator.manual_seed(seed_and_pipeline_args["seed"])
    session.set_pipeline_args(**seed_and_pipeline_args["pipeline_args"])


@app.task
def update_prompt():
    session.update_prompt()


@app.task
def update_conditioning():
    session.update_conditioning()


@app.task
def generate():
    video = session.generate()
    video_np = video.permute(1, 2, 3, 0).cpu().float().numpy()
    video_np = (video_np * 255).astype(np.uint8)
    return video_np
