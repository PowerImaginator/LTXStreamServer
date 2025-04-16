import sys

# Needed to support relative imports? (e.g. LTXVideoSession)
sys.path.append(".")

import io
import random
import asyncio
import atexit
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import imageio
import msgpack
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from LTXVideoSession import LTXVideoSession


# https://stackoverflow.com/a/56944547
async def multiprocessing_queue_get_async(queue: multiprocessing.Queue):
    executor = ProcessPoolExecutor(max_workers=1)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, queue.get)


@atexit.register
def kill_children():
    [p.kill() for p in multiprocessing.active_children()]


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


class SessionProcess:
    def __init__(self):
        manager = multiprocessing.Manager()
        self.receive_queue = manager.Queue()
        self.send_queue = manager.Queue()

        self.device = None
        self.generator = None
        self.session = None

        self._process = multiprocessing.Process(
            target=self._run,
            daemon=True,
        )
        self._process.start()

    def terminate(self):
        self.send_queue.put({"type": "__INFERENCE_SERVER_TERMINATE__"})
        self._process.terminate()
        self._process = None
        self.device = None
        self.generator = None
        self.session = None

    def receive_message(self, item):
        self.receive_queue.put(item)

    def _run(self):
        while True:
            message = self.receive_queue.get()

            if message["type"] == "CREATE_PIPELINE":
                result = self._create_pipeline(message)
            elif message["type"] == "SET_PIPELINE_ARGS":
                result = self._set_pipeline_args(message)
            elif message["type"] == "UPDATE_PROMPT":
                result = self._update_prompt(message)
            elif message["type"] == "UPDATE_CONDITIONING":
                result = self._update_conditioning(message)
            elif message["type"] == "GENERATE":
                result = self._generate(message)
            else:
                result = {
                    "type": "ERROR",
                    "error": "Unrecognized message type: {}".format(message["type"]),
                }

            self.send_queue.put(result)

    def _create_pipeline(self, message):
        if self.device is None:
            self.device = get_device()
        if self.generator is None:
            self.generator = torch.Generator(device=self.device)
        if self.session is None:
            self.session = LTXVideoSession(
                ckpt_path="checkpoints/ltx-video-2b-v0.9.5.safetensors",
                precision="bfloat16",
                text_encoder_model_name_or_path="PixArt-alpha/PixArt-XL-2-1024-MS",
                device=self.device,
                enhance_prompt=False,
                prompt_enhancer_image_caption_model_name_or_path=None,
                prompt_enhancer_llm_model_name_or_path=None,
            )
        return {"type": "READY"}

    def _set_pipeline_args(self, message):
        if "seed" in message:
            seed_everything(message["seed"])
            self.generator.manual_seed(message["seed"])
        self.session.set_pipeline_args(**message["pipeline_args"])
        return {"type": "READY"}

    def _update_prompt(self, message):
        self.session.update_prompt()
        return {"type": "READY"}

    def _update_conditioning(self, message):
        self.session.update_conditioning()
        return {"type": "READY"}

    def _generate(self, message):
        video = self.session.generate()
        video_np = video.permute(1, 2, 3, 0).cpu().float().numpy()
        video_np = (video_np * 255).astype(np.uint8)
        out_io = io.BytesIO()
        with imageio.get_writer(
            out_io,
            format="mp4",
            quality=10,
            fps=self.session.pipeline_args["frame_rate"],
        ) as video:
            for frame in video_np:
                video.append_data(frame)

        return {
            "type": "OUTPUT",
            "video_bytes": out_io.getvalue(),
        }


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def send_message_thread_task(
    websocket: WebSocket, session_process: SessionProcess
):
    while True:
        message = await multiprocessing_queue_get_async(session_process.send_queue)
        if message["type"] == "__INFERENCE_SERVER_TERMINATE__":
            break
        else:
            await websocket.send_bytes(msgpack.packb(message))


# https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task
background_tasks = set()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    session_process = SessionProcess()

    task = asyncio.create_task(send_message_thread_task(websocket, session_process))
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)

    try:
        while True:
            received_message = msgpack.unpackb(await websocket.receive_bytes())
            session_process.receive_message(received_message)
    except WebSocketDisconnect:
        pass

    session_process.terminate()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
