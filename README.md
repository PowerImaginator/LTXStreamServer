# LTXStream Server

This server is intended for use with the LTXStream Client application: https://github.com/PowerImaginator/LTXStreamClient

## Setup

First, install Python >= 3.10 and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```sh
git clone https://github.com/PowerImaginator/LTXStreamServer.git
cd LTXStreamServer
uv sync
mkdir checkpoints && cd checkpoints
wget https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltx-video-2b-v0.9.5.safetensors
cd ..
```

## Run

```sh
uv run fastapi run inference-server.py
```
