import modal

APP_NAME = "omnitry"
REPO_DIR = "/root/OmniTry"
CHECKPOINT_DIR = f"{REPO_DIR}/checkpoints"

volume = modal.Volume.from_name("omnitry-weights-vol", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install(
        "git", "ffmpeg", "libgl1", "libglib2.0-0", "libsm6",
        "libxext6", "libxrender1", "build-essential", "ninja-build"
    )
    .run_commands("pip install --upgrade pip setuptools wheel packaging")
    # torch first — everything depends on this.
    # NOTE: flash-attn is intentionally NOT installed.
    #   - flash-attn 2.7.4.post1 has a known ABI break with torch 2.7.0 (undefined symbol at runtime)
    #   - flash-attn 2.8.3 official Linux prebuilt wheels only cover up to torch2.5
    #   - PyTorch 2.7 ships SDPA with built-in FlashAttention-2 kernels via cuDNN
    #   - diffusers falls back to SDPA automatically, giving ~80% of flash-attn's benefit
    .pip_install(
        "torch==2.7.0", "torchvision==0.22.0", "triton==3.3.0",
        extra_options="--index-url https://download.pytorch.org/whl/cu126"
    )
    .pip_install(
        "accelerate==1.7.0", "diffusers==0.33.1", "transformers==4.45.0",
        "gradio==5.6.0", "gradio-client==1.4.3", "huggingface-hub==0.32.1",
        "peft==0.13.2", "sentencepiece==0.2.0", "safetensors==0.5.3",
        "einops==0.8.1", "omegaconf==2.3.0", "numpy==2.2.6", "pandas==2.2.3",
        "pillow==11.2.1", "protobuf==3.20.3", "scipy", "tqdm==4.67.1",
        "requests==2.32.3", "fastapi==0.115.12", "uvicorn==0.34.2",
        "pydantic==2.9.2", "torchmetrics", "lpips", "supervision",
        "psutil", "pyyaml", "regex"
    )
    .run_commands(
        f"git clone https://github.com/deviltoonix/OmniTry.git {REPO_DIR}",
        f"mkdir -p {CHECKPOINT_DIR}"
    )
    .env({
        "HF_HOME": "/root/hf_cache",
        "TRANSFORMERS_CACHE": "/root/hf_cache",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTHONUNBUFFERED": "1",
        "GRADIO_SERVER_NAME": "0.0.0.0",
        "GRADIO_SERVER_PORT": "8000",
        # Use SDPA (built-in to torch 2.7, uses FA2 kernels via cuDNN on A100)
        "DIFFUSERS_ATTN_IMPLEMENTATION": "sdpa",
        "TORCHDYNAMO_VERBOSE": "0",
    })
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600,
    max_containers=1,
    scaledown_window=120,   # Keep alive 2min so torch.compile cache survives between requests
    volumes={CHECKPOINT_DIR: volume},
    secrets=[modal.Secret.from_name("hf-secret")],
)
@modal.asgi_app()
def run():
    import os
    import sys
    import torch
    import runpy
    from unittest.mock import MagicMock
    import gradio as gr
    from fastapi import FastAPI

    print(f"Torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    # Confirm SDPA flash kernel is available (should be True on A100 with torch 2.7)
    print(f"SDPA flash kernel available: {torch.backends.cuda.flash_sdp_enabled()}")

    os.chdir(REPO_DIR)
    sys.path.insert(0, REPO_DIR)
    os.makedirs(os.path.join(REPO_DIR, ".gradio"), exist_ok=True)

    launch_kwargs = {}
    def mock_launch(*args, **kwargs):
        launch_kwargs.update(kwargs)
        return None
    gr.Blocks.launch = MagicMock(side_effect=mock_launch)

    print("Executing gradio_demo.py...")
    script_vars = runpy.run_path(
        os.path.join(REPO_DIR, "gradio_demo.py"),
        run_name="__main__"
    )

    if "demo" in script_vars:
        demo_app = script_vars["demo"]
    else:
        demo_app = next(
            (val for val in script_vars.values() if isinstance(val, gr.Blocks)),
            None
        )
    if not demo_app:
        raise ValueError("Could not find demo object in gradio_demo.py")

    demo_app.max_file_size = launch_kwargs.get("max_file_size", None)
    demo_app.queue(default_concurrency_limit=1)

    if "allowed_paths" in launch_kwargs:
        if not hasattr(demo_app, "allowed_paths"):
            demo_app.allowed_paths = []
        demo_app.allowed_paths.extend(launch_kwargs["allowed_paths"])

    web_app = FastAPI()
    web_app = gr.mount_gradio_app(web_app, demo_app, path="/")
    return web_app
