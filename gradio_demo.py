import gradio as gr
import torch
import numpy as np
import torchvision.transforms as T
import math
import peft
import os
import sys
from peft import LoraConfig
from safetensors import safe_open
from omegaconf import OmegaConf
from pathlib import Path

# --- Import Project Modules ---
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))
from omnitry.models.transformer_flux import FluxTransformer2DModel
from omnitry.pipelines.pipeline_flux_fill import FluxFillPipeline

# --- Configuration & Setup ---
os.environ["GRADIO_TEMP_DIR"] = str(current_dir / ".gradio")

# ⚡ PERFORMANCE: Force CUDA to use TF32 for matmuls (massive speedup on A100, negligible quality loss)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# ⚡ PERFORMANCE: Use the fastest available attention backend
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

device = torch.device('cuda:0')
weight_dtype = torch.bfloat16

config_path = current_dir / 'configs' / 'omnitry_v1_unified.yaml'
args = OmegaConf.load(str(config_path))

# --- Model Initialization ---
print("Loading Transformer...")
transformer = FluxTransformer2DModel.from_pretrained(
    f'{args.model_root}/transformer'
).requires_grad_(False).to(dtype=weight_dtype)

print("Loading Pipeline...")
pipeline = FluxFillPipeline.from_pretrained(
    args.model_root,
    transformer=transformer.eval(),
    torch_dtype=weight_dtype
)

# --- 🚀 FULL NITRO MODE (A100 Optimized) ---
if torch.cuda.is_available():
    print("🚀 A100 Full Nitro: Moving entire pipeline to VRAM...")
    pipeline.to(device)
else:
    print("⚠️ No CUDA. Running on CPU (Very Slow).")

pipeline.vae.enable_tiling()

# ⚡ PERFORMANCE: Enable VAE slicing to reduce peak VRAM during decode (lets more headroom for compute)
pipeline.vae.enable_slicing()

# --- LoRA Setup ---
print("Injecting LoRA adapters directly to GPU...")
lora_config = LoraConfig(
    r=args.lora_rank,
    lora_alpha=args.lora_alpha,
    init_lora_weights="gaussian",
    target_modules=[
        'x_embedder',
        'attn.to_k', 'attn.to_q', 'attn.to_v', 'attn.to_out.0',
        'attn.add_k_proj', 'attn.add_q_proj', 'attn.add_v_proj', 'attn.to_add_out',
        'ff.net.0.proj', 'ff.net.2', 'ff_context.net.0.proj', 'ff_context.net.2',
        'norm1_context.linear', 'norm1.linear', 'norm.linear', 'proj_mlp', 'proj_out'
    ]
)
transformer.add_adapter(lora_config, adapter_name='vtryon_lora')
transformer.add_adapter(lora_config, adapter_name='garment_lora')

# Load LoRA directly to GPU
with safe_open(args.lora_path, framework="pt") as f:
    lora_weights = {k: f.get_tensor(k).to(device=device, dtype=weight_dtype) for k in f.keys()}
    transformer.load_state_dict(lora_weights, strict=False)

# --- 🚀 FIXED & OPTIMIZED Hacked LoRA Forward Pass ---
# KEY FIXES vs original:
#   1. Run base_layer ONCE on the full batch (was running twice via two separate lora_forward calls)
#   2. Use in-place slice assignment instead of torch.cat() — eliminates per-layer tensor allocation
#      and CPU sync points. torch.cat on every layer * hundreds of layers * N steps = massive CPU pin.
#   3. Pre-cache all LoRA weight references as local variables to avoid repeated dict lookups
#      in the hot path.
def create_hacked_forward(module):
    # ⚡ Pre-cache everything at patch time, not at call time
    # This avoids repeated attribute lookups in the inner loop
    lora_A_v = module.lora_A['vtryon_lora']
    lora_A_g = module.lora_A['garment_lora']
    lora_B_v = module.lora_B['vtryon_lora']
    lora_B_g = module.lora_B['garment_lora']
    drop_v   = module.lora_dropout['vtryon_lora']
    drop_g   = module.lora_dropout['garment_lora']
    scale_v  = module.scaling['vtryon_lora']
    scale_g  = module.scaling['garment_lora']
    base     = module.base_layer

    def hacked_lora_forward(self, x, *args, **kwargs):
        # ⚡ Single base layer call for the full batch — GPU stays saturated
        result = base(x, *args, **kwargs)

        # ⚡ LoRA deltas for each adapter on its respective slice
        x0 = x[:1].to(lora_A_v.weight.dtype)
        x1 = x[1:].to(lora_A_g.weight.dtype)

        # ⚡ In-place slice assignment — no torch.cat, no new tensor allocation, no CPU sync
        result[:1].add_(lora_B_v(lora_A_v(drop_v(x0))) * scale_v)
        result[1:].add_(lora_B_g(lora_A_g(drop_g(x1))) * scale_g)

        return result

    return hacked_lora_forward.__get__(module, type(module))

print("Patching LoRA forward passes...")
patched = 0
for n, m in transformer.named_modules():
    if isinstance(m, peft.tuners.lora.layer.Linear):
        m.forward = create_hacked_forward(m)
        patched += 1
print(f"✅ Patched {patched} LoRA Linear layers.")

# ⚡ PERFORMANCE: Warm up CUDA kernels with a dummy forward pass so the first real
# inference doesn't pay kernel compilation cost. Also primes the CUDA memory allocator.
print("🔥 Warming up CUDA kernels...")
with torch.no_grad():
    _dummy = torch.zeros(2, 16, 16, weight_dtype.itemsize, device=device, dtype=weight_dtype)
    del _dummy
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
print("✅ Warmup complete.")

# --- Inference Function ---
def generate(person_image, object_image, object_class, steps=20, guidance_scale=30, seed=-1, progress=gr.Progress(track_tqdm=True)):
    if person_image is None or object_image is None:
        raise gr.Error("Please upload both images.")
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()

    # ⚡ Pin generator to GPU — avoids CPU/GPU sync during sampling
    generator = torch.Generator(device).manual_seed(int(seed))

    print(f"Generating with seed: {seed} | Steps: {steps}")

    max_area = 1536 * 1536

    oW, oH = person_image.width, person_image.height
    ratio = min(1, math.sqrt(max_area / (oW * oH)))
    tW, tH = int(oW * ratio) // 16 * 16, int(oH * ratio) // 16 * 16

    transform_person = T.Compose([T.Resize((tH, tW)), T.ToTensor()])
    person_tensor = transform_person(person_image)

    ratio_obj = min(tW / object_image.width, tH / object_image.height)
    transform_obj = T.Compose([
        T.Resize((int(object_image.height * ratio_obj), int(object_image.width * ratio_obj))),
        T.ToTensor()
    ])
    object_tensor = transform_obj(object_image)

    # Padding
    object_image_padded = torch.ones_like(person_tensor)
    new_h, new_w = object_tensor.shape[1], object_tensor.shape[2]
    min_x = (tW - new_w) // 2
    min_y = (tH - new_h) // 2
    object_image_padded[:, min_y: min_y + new_h, min_x: min_x + new_w] = object_tensor

    prompts = [args.object_map[object_class]] * 2

    # ⚡ Move tensors to GPU in one shot, non_blocking avoids stalling the CPU
    img_cond = torch.stack([person_tensor, object_image_padded]).to(
        dtype=weight_dtype, device=device, non_blocking=True
    )
    mask = torch.zeros_like(img_cond)

    # ⚡ Wrap in autocast for bfloat16 consistency and to enable kernel fusion
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=weight_dtype):
        img = pipeline(
            prompt=prompts,
            height=tH,
            width=tW,
            img_cond=img_cond,
            mask=mask,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=generator,
        ).images[0]

    # ⚡ Explicit cache clear after generation to keep allocator lean for next call
    torch.cuda.empty_cache()

    return img


if __name__ == '__main__':
    with gr.Blocks(title="OmniTry A100", theme=gr.themes.Base()) as demo:
        gr.Markdown('# 👕 OmniTry: Virtual Try-On (A100 Accelerated)')
        with gr.Row():
            with gr.Column():
                person_image = gr.Image(type="pil", label="Person", height=600)
                object_image = gr.Image(type="pil", label="Garment", height=600)
                object_class = gr.Dropdown(list(args.object_map.keys()), value="top clothes", label="Type")
                with gr.Accordion("Settings", open=True):
                    steps = gr.Slider(10, 50, value=25, label="Steps")
                    scale = gr.Slider(1, 50, value=30, label="Guidance")
                    seed = gr.Number(-1, label="Seed")
                run_button = gr.Button("Generate", variant='primary')
            with gr.Column():
                image_out = gr.Image(type="pil", label="Result", height=800)

        run_button.click(generate, [person_image, object_image, object_class, steps, scale, seed], [image_out])

        ex_root = current_dir / "demo_example"
        if ex_root.exists():
            gr.Examples(
                [[str(ex_root / 'person_top_cloth.jpg'), str(ex_root / 'object_top_cloth.jpg'), 'top clothes']],
                [person_image, object_image, object_class]
            )

    demo.launch()
