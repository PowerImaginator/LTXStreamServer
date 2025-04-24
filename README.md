# LTXStream Server

More documentation will be written once this application is fully functional.

```sh
uv venv
source .venv/bin/activate
uv sync

mkdir checkpoints
cd checkpoints
wget https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltxv-2b-0.9.6-distilled-04-25.safetensors?download=true
cd ..

mkdir outputs

python playground.py
```
