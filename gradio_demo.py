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
# We do NOT use cpu_offload. We force everything to the 80GB GPU.
if torch.cuda.is_available():
    print("🚀 A100 Full Nitro: Moving entire pipeline to VRAM...")
    pipeline.to(device)
else:
    print("⚠️ No CUDA. Running on CPU (Very Slow).")

pipeline.vae.enable_tiling()

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

# Load LoRA directly to GPU to avoid CPU bottlenecks
with safe_open(args.lora_path, framework="pt") as f:
    lora_weights = {k: f.get_tensor(k).to(device=device, dtype=weight_dtype) for k in f.keys()}
    transformer.load_state_dict(lora_weights, strict=False)

# --- Hacked LoRA Forward Pass ---
def create_hacked_forward(module):
    def lora_forward(self, active_adapter, x, *args, **kwargs):
        result = self.base_layer(x, *args, **kwargs)
        if active_adapter is not None:
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            x = x.to(lora_A.weight.dtype)
            result = result + lora_B(lora_A(dropout(x))) * scaling
        return result
    
    def hacked_lora_forward(self, x, *args, **kwargs):
        return torch.cat((
            lora_forward(self, 'vtryon_lora', x[:1], *args, **kwargs),
            lora_forward(self, 'garment_lora', x[1:], *args, **kwargs),
        ), dim=0)
    
    return hacked_lora_forward.__get__(module, type(module))

for n, m in transformer.named_modules():
    if isinstance(m, peft.tuners.lora.layer.Linear):
        m.forward = create_hacked_forward(m)


# --- Inference Function ---
def generate(person_image, object_image, object_class, steps=20, guidance_scale=30, seed=-1, progress=gr.Progress(track_tqdm=True)):
    if person_image is None or object_image is None:
        raise gr.Error("Please upload both images.")

    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    generator = torch.Generator(device).manual_seed(int(seed))
    
    print(f"Generating with seed: {seed} | Steps: {steps}")

    # 🚀 Resolution: 1536x1536 (Fast & Sharp on A100)
    max_area = 1536 * 1536 
    
    oW, oH = person_image.width, person_image.height
    ratio = min(1, math.sqrt(max_area / (oW * oH)))
    tW, tH = int(oW * ratio) // 16 * 16, int(oH * ratio) // 16 * 16
    
    transform_person = T.Compose([T.Resize((tH, tW)), T.ToTensor()])
    person_tensor = transform_person(person_image)

    # Object Resize
    ratio_obj = min(tW / object_image.width, tH / object_image.height)
    transform_obj = T.Compose([T.Resize((int(object_image.height * ratio_obj), int(object_image.width * ratio_obj))), T.ToTensor()])
    object_tensor = transform_obj(object_image)
    
    # Padding
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
            prompt=prompts, height=tH, width=tW, img_cond=img_cond, mask=mask,
            guidance_scale=guidance_scale, num_inference_steps=steps, generator=generator,
        ).images[0]

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
            gr.Examples([[str(ex_root/'person_top_cloth.jpg'), str(ex_root/'object_top_cloth.jpg'), 'top clothes']], [person_image, object_image, object_class])
    
    demo.launch()
