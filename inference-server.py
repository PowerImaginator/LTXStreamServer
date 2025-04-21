import sys

# Needed to support relative imports? (e.g. LTXVideoSession)
sys.path.append(".")

import asyncio
import msgpack
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import tasks

import io  # TODO: Remove
import imageio  # TODO: Remove

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def async_get_result(task):
    return await asyncio.get_running_loop().run_in_executor(None, task.get)


def encode_generated_video(video_np):
    out_io = io.BytesIO()
    with imageio.get_writer(
        out_io,
        format="mp4",
        # quality=10,
        fps=25,  # TODO: Get this from tasks.py since it will remember last time set_pipeline_args was called with a framerate
    ) as video:
        for frame in video_np:
            video.append_data(frame)
    return {
        "type": "OUTPUT",
        "video_bytes": out_io.getvalue(),
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            received_message = msgpack.unpackb(await websocket.receive_bytes())

            if received_message["type"] == "CREATE_PIPELINE":
                if not await async_get_result(tasks.has_pipeline.delay()):
                    await async_get_result(tasks.create_pipeline.delay())
                await websocket.send_bytes(msgpack.packb({"type": "READY"}))
            elif received_message["type"] == "SET_PIPELINE_ARGS":
                await async_get_result(
                    tasks.set_pipeline_args.delay(
                        {
                            "seed": received_message["seed"],
                            "pipeline_args": received_message["pipeline_args"],
                        }
                    )
                )
                await websocket.send_bytes(msgpack.packb({"type": "READY"}))
            elif received_message["type"] == "UPDATE_PROMPT":
                await async_get_result(tasks.update_prompt.delay())
                await websocket.send_bytes(msgpack.packb({"type": "READY"}))
            elif received_message["type"] == "UPDATE_CONDITIONING":
                await async_get_result(tasks.update_conditioning.delay())
                await websocket.send_bytes(msgpack.packb({"type": "READY"}))
            elif received_message["type"] == "GENERATE":
                result = await async_get_result(tasks.generate.delay())
                await websocket.send_bytes(
                    msgpack.packb(encode_generated_video(result))
                )
            else:
                await websocket.send_bytes(
                    msgpack.packb(
                        {
                            "type": "ERROR",
                            "error": "Unrecognized message type: "
                            + received_message["type"],
                        }
                    )
                )
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
