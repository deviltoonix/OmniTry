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
# Ensure the current directory is in python path to find 'omnitry'
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from omnitry.models.transformer_flux import FluxTransformer2DModel
from omnitry.pipelines.pipeline_flux_fill import FluxFillPipeline

# --- Configuration & Setup ---
os.environ["GRADIO_TEMP_DIR"] = str(current_dir / ".gradio")
device = torch.device('cuda:0')
weight_dtype = torch.bfloat16

# Robust config loading (handles running from different dirs)
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

# --- 🚀 OPTIMIZATION: Smart VRAM Management ---
# The L40S has 48GB VRAM. The model peaks at ~26GB.
# Offloading to CPU slows things down massively. We only offload if VRAM is tight.
if torch.cuda.is_available():
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Detected VRAM: {total_vram_gb:.1f} GB")

    if total_vram_gb > 40:
        print("✨ High VRAM detected (L40S/A6000). Keeping model on GPU for maximum speed.")
        pipeline.to(device)
    else:
        print("⚠️ Limited VRAM. Enabling CPU offload (Slower but saves memory).")
        pipeline.enable_model_cpu_offload()
else:
    print("⚠️ No CUDA detected. Running on CPU (Will be very slow).")

pipeline.vae.enable_tiling()


# --- LoRA Setup ---
print("Injecting LoRA adapters...")
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

with safe_open(args.lora_path, framework="pt") as f:
    lora_weights = {k: f.get_tensor(k) for k in f.keys()}
    transformer.load_state_dict(lora_weights, strict=False)


# --- Hacked LoRA Forward Pass ---
# Keeps the original logic but wrapped cleanly
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
        # Forward pass splitting logic
        return torch.cat((
            lora_forward(self, 'vtryon_lora', x[:1], *args, **kwargs),
            lora_forward(self, 'garment_lora', x[1:], *args, **kwargs),
        ), dim=0)
    
    return hacked_lora_forward.__get__(module, type(module))

# Apply the hack
for n, m in transformer.named_modules():
    if isinstance(m, peft.tuners.lora.layer.Linear):
        m.forward = create_hacked_forward(m)


# --- Inference Function ---
def generate(person_image, object_image, object_class, steps=20, guidance_scale=30, seed=-1, progress=gr.Progress(track_tqdm=True)):
    if person_image is None or object_image is None:
        raise gr.Error("Please upload both a Person Image and an Object Image.")

    # Handle Seed locally (Better for concurrency)
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    generator = torch.Generator(device).manual_seed(int(seed))
    
    print(f"Generating with seed: {seed} | Steps: {steps} | Scale: {guidance_scale}")

    # Resize logic (Person)
    # 🚀 QUALITY BOOST: Increased from 1024x1024 to 1536x1536 (2.25x pixels)
    # Your L40S has plenty of VRAM for this.
    max_area = 1280 * 1280 
    
    oW, oH = person_image.width, person_image.height
    ratio = min(1, math.sqrt(max_area / (oW * oH)))
    tW, tH = int(oW * ratio) // 16 * 16, int(oH * ratio) // 16 * 16
    
    transform_person = T.Compose([
        T.Resize((tH, tW)),
        T.ToTensor(),
    ])
    person_tensor = transform_person(person_image)

    # Resize logic (Object / Garment)
    ratio_obj = min(tW / object_image.width, tH / object_image.height)
    transform_obj = T.Compose([
        T.Resize((int(object_image.height * ratio_obj), int(object_image.width * ratio_obj))),
        T.ToTensor(),
    ])
    
    object_tensor = transform_obj(object_image)
    
    # Padding
    object_image_padded = torch.ones_like(person_tensor)
    new_h, new_w = object_tensor.shape[1], object_tensor.shape[2]
    min_x = (tW - new_w) // 2
    min_y = (tH - new_h) // 2
    object_image_padded[:, min_y: min_y + new_h, min_x: min_x + new_w] = object_tensor

    # Prepare batch
    prompts = [args.object_map[object_class]] * 2
    img_cond = torch.stack([person_tensor, object_image_padded]).to(dtype=weight_dtype, device=device) 
    mask = torch.zeros_like(img_cond).to(img_cond)

    with torch.no_grad():
        img = pipeline(
            prompt=prompts,
            height=tH,
            width=tW,     
            img_cond=img_cond,
            mask=mask,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=generator, # Use local generator
        ).images[0]

    return img


# --- Gradio UI ---
if __name__ == '__main__':
    
    # Define styles or custom CSS here if needed
    with gr.Blocks(title="OmniTry Demo", theme=gr.themes.Base()) as demo:
        gr.Markdown('# 👕 OmniTry: Virtual Try-On Demo')
        gr.Markdown('Upload a person and a garment to generate a virtual try-on result.')
        
        with gr.Row():
            with gr.Column():
                person_image = gr.Image(type="pil", label="Person Image", sources=['upload', 'clipboard'], height=600)
                object_image = gr.Image(type="pil", label="Garment Image", sources=['upload', 'clipboard'], height=600)
                object_class = gr.Dropdown(label='Garment Type', choices=list(args.object_map.keys()), value="top clothes")
                
                with gr.Accordion("Advanced Settings", open=False):
                    guidance_scale = gr.Slider(label="Guidance Scale", minimum=1, maximum=50, value=30, step=0.1)
                    steps = gr.Slider(label="Inference Steps", minimum=1, maximum=50, value=20, step=1)
                    seed = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                
                run_button = gr.Button(value="✨ Generate Try-On", variant='primary', size="lg")

            with gr.Column():
                image_out = gr.Image(type="pil", label="Result", height=800, interactive=False)

        # Connect the button
        run_button.click(
            fn=generate, 
            inputs=[person_image, object_image, object_class, steps, guidance_scale, seed], 
            outputs=[image_out]
        )

        # Add Examples (Paths must exist in the container)
        # Verify paths before adding to prevent broken UI
        example_root = current_dir / "demo_example"
        if example_root.exists():
            gr.Examples(
                examples=[
                    [str(example_root/'person_top_cloth.jpg'), str(example_root/'object_top_cloth.jpg'), 'top clothes'],
                    [str(example_root/'person_dress.jpg'), str(example_root/'object_dress.jpg'), 'dress'],
                ],
                inputs=[person_image, object_image, object_class],
                label="Quick Examples"
            )
    
    demo.launch()
