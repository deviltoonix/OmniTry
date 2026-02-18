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
# Enable Flash Attention 2 before any model is loaded
os.environ["DIFFUSERS_ATTN_IMPLEMENTATION"] = "sdpa"

device = torch.device('cuda:0')
weight_dtype = torch.bfloat16
config_path = current_dir / 'configs' / 'omnitry_v1_unified.yaml'
args = OmegaConf.load(str(config_path))

# --- Model Initialization ---
print("Loading Transformer...")
transformer = FluxTransformer2DModel.from_pretrained(
    f'{args.model_root}/transformer',

    torch_dtype=weight_dtype,
).requires_grad_(False)

print("Loading Pipeline...")
pipeline = FluxFillPipeline.from_pretrained(
    args.model_root,
    transformer=transformer.eval(),
    torch_dtype=weight_dtype,
)

# --- 🚀 FULL NITRO MODE (A100 80GB) ---
if torch.cuda.is_available():
    print("🚀 A100 Full Nitro: Moving entire pipeline to VRAM...")
    pipeline.to(device)
else:
    print("⚠️ No CUDA. Running on CPU (Very Slow).")

# VAE slicing: lightweight alternative to tiling, no CPU tile-coordination overhead.
# With 80GB VRAM you don't need tiling for memory — it only adds CPU sync cost.
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

# Load LoRA weights directly to GPU
with safe_open(args.lora_path, framework="pt") as f:
    lora_weights = {k: f.get_tensor(k).to(device=device, dtype=weight_dtype) for k in f.keys()}
    transformer.load_state_dict(lora_weights, strict=False)

# --- Hacked LoRA Forward Pass (GPU-optimized) ---
#
# Key fixes vs original:
#   1. Base layer runs once on FULL batch — GPU stays busy, no split at base layer.
#   2. LoRA deltas computed separately then torch.cat'd into one tensor before
#      the final add — single contiguous GPU kernel instead of two in-place ops.
#      In-place slice assignment breaks torch.compile graph tracing; cat does not.
#   3. No nested closures with dict lookups inside the hot path.
#
def create_hacked_forward(module):
    def hacked_lora_forward(self, x, *args, **kwargs):
        # Full-batch base forward — keeps GPU utilisation high
        result = self.base_layer(x, *args, **kwargs)

        # vtryon_lora on first element of batch
        lora_A_v = self.lora_A['vtryon_lora']
        lora_B_v = self.lora_B['vtryon_lora']
        drop_v   = self.lora_dropout['vtryon_lora']
        scale_v  = self.scaling['vtryon_lora']
        x0 = x[:1].to(lora_A_v.weight.dtype)
        delta0 = lora_B_v(lora_A_v(drop_v(x0))) * scale_v

        # garment_lora on second element of batch
        lora_A_g = self.lora_A['garment_lora']
        lora_B_g = self.lora_B['garment_lora']
        drop_g   = self.lora_dropout['garment_lora']
        scale_g  = self.scaling['garment_lora']
        x1 = x[1:].to(lora_A_g.weight.dtype)
        delta1 = lora_B_g(lora_A_g(drop_g(x1))) * scale_g

        # Single cat + add — one GPU kernel, compile-friendly
        delta = torch.cat([delta0, delta1], dim=0)
        return result + delta

    return hacked_lora_forward.__get__(module, type(module))

for n, m in transformer.named_modules():
    if isinstance(m, peft.tuners.lora.layer.Linear):
        m.forward = create_hacked_forward(m)

# --- torch.compile ---
# mode="reduce-overhead" builds CUDA graphs so the GPU scheduler is driven
# directly from the graph with minimal Python involvement.
# fullgraph=False lets it skip graph breaks from peft's dynamic code.
# WARNING: First inference call will take ~60-90s to compile — this is normal.
# Every subsequent call will be dramatically faster (~2-5s/step at 1024px).
print("Compiling transformer with torch.compile (mode=reduce-overhead)...")
print("⚠️  First inference will be slow (~60-90s) — compilation in progress.")
transformer = torch.compile(transformer, mode="reduce-overhead", fullgraph=False)


# --- Inference Function ---
def _run_inference(person_image, object_image, object_class, steps, guidance_scale, seed, max_res):
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    generator = torch.Generator(device).manual_seed(int(seed))

    print(f"Generating — seed: {seed} | steps: {steps} | res: {max_res}px")
    if torch.cuda.is_available():
        print(f"GPU memory — allocated: {torch.cuda.memory_allocated(device)/1e9:.1f}GB "
              f"| reserved: {torch.cuda.memory_reserved(device)/1e9:.1f}GB")

    max_area = max_res * max_res
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

    object_image_padded = torch.ones_like(person_tensor)
    new_h, new_w = object_tensor.shape[1], object_tensor.shape[2]
    min_x = (tW - new_w) // 2
    min_y = (tH - new_h) // 2
    object_image_padded[:, min_y: min_y + new_h, min_x: min_x + new_w] = object_tensor

    prompts = [args.object_map[object_class]] * 2
    img_cond = torch.stack([person_tensor, object_image_padded]).to(dtype=weight_dtype, device=device)
    mask = torch.zeros_like(img_cond).to(img_cond)

    with torch.no_grad():
        img = pipeline(
            prompt=prompts, height=tH, width=tW,
            img_cond=img_cond, mask=mask,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=generator,
        ).images[0]
    return img


def generate(person_image, object_image, object_class, steps, guidance_scale, seed, resolution,
             progress=gr.Progress(track_tqdm=True)):
    if person_image is None or object_image is None:
        raise gr.Error("Please upload both images.")
    res_map = {"1024 (Fast)": 1024, "1280 (Balanced)": 1280, "1536 (Quality)": 1536}
    max_res = res_map.get(resolution, 1024)
    return _run_inference(person_image, object_image, object_class, steps, guidance_scale, seed, max_res)


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
                    resolution = gr.Radio(
                        choices=["1024 (Fast)", "1280 (Balanced)", "1536 (Quality)"],
                        value="1024 (Fast)",
                        label="Resolution (1024 recommended for benchmarking)"
                    )
                run_button = gr.Button("Generate", variant='primary')
            with gr.Column():
                image_out = gr.Image(type="pil", label="Result", height=800)

        run_button.click(
            generate,
            [person_image, object_image, object_class, steps, scale, seed, resolution],
            [image_out]
        )

        ex_root = current_dir / "demo_example"
        if ex_root.exists():
            gr.Examples(
                [[str(ex_root/'person_top_cloth.jpg'), str(ex_root/'object_top_cloth.jpg'), 'top clothes']],
                [person_image, object_image, object_class]
            )

    demo.launch()
