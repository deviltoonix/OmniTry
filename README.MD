<p align="center">
<h2 align="center">OmniTry: Virtual Try-On Anything without Masks</h2>
<p align="center">
<a href="https://scholar.google.com.hk/citations?user=mZwJLeUAAAAJ&hl=zh-CN"><strong>Yutong Feng</strong></a> ·
<a href=""><strong>Linlin Zhang</strong></a> ·
<a href=""><strong>Hengyuan Cao</strong></a> ·
<a href="https://scholar.google.com.hk/citations?hl=zh-CN&user=LxiMyjQAAAAJ"><strong>Yiming Chen</strong></a> ·
<a href=""><strong>Xiaoduan Feng</strong></a> ·
<a href=""><strong>Jian Cao</strong></a> ·
<a href=""><strong>Yuxiong Wu</strong></a> ·
<a href="https://scholar.google.com.hk/citations?user=6hTbqDEAAAAJ&hl=zh-CN"><strong>Bin Wang</strong></a>



<b>Kunbyte AI &nbsp; | &nbsp; Zhejiang University </b>




<a href="http://arxiv.org/abs/2508.13632"><img src='https://img.shields.io/badge/arXiv-OmniTry-red' alt='Paper PDF'></a>
<a href="https://omnitry.github.io/"><img src='https://img.shields.io/badge/project page-OmniTry-green'></a>
<a href='https://huggingface.co/Kunbyte/OmniTry'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow'></a>
<a href='https://huggingface.co/spaces/Kunbyte/OmniTry'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>



</p>
</p>

🚀 Easy Cloud Deployment (Modal)
Want to run OmniTry in the cloud without setting up a complex computer? Follow these 5 simple steps.

Step 1: Get a Modal Account
Go to Modal.com and sign up. They usually give you some free credits to start!

Step 2: Install the Modal Tool
Open the "Terminal" on your computer and type this:
  pip install modal

Step 3: Link Your Account
Tell your computer who you are on Modal by typing:

  modal setup
A window will pop up in your internet browser. Just click "Login".

Step 4: Add Your Secret Key
Since we use models from Hugging Face, you need a "Secret Key" so Modal can download them for you.

Get your token from Hugging Face Settings.

Type this in your terminal:

modal secret create hf-secret HF_TOKEN=your_token_here
(Replace your_token_here with the long code from Hugging Face)

Step 5: Launch the Demo! 🚀
Now, just run the deployment script. This will set up the GPU, download the weights, and give you a website link to play with:

modal deploy modal_deploy.py
That's it! Check the terminal for a link that ends in .modal.run. Click it to start your virtual try-on!

💻 Local Get Started
If you have a powerful computer at home (at least 28GB VRAM), you can run it locally.

1. Download Weights
Create a folder: mkdir checkpoints

Put the FLUX.1-Fill-dev files in checkpoints/FLUX.1-Fill-dev

Put the OmniTry LoRA in checkpoints/omnitry_v1_unified.safetensors

2. Prepare Environment
Bash
pip install -r requirements.txt
# Optional: Make it faster!
pip install flash-attn --no-build-isolation
3. Start the Demo
Bash
python gradio_demo.py
📜 Acknowledgements
Built with love using diffusers and FLUX.

✍️ Citation
Code snippet
@article{feng2025omnitry,
  title={OmniTry: Virtual Try-On Anything without Masks},
  author={Feng, Yutong and Zhang, Linlin and Cao, Hengyuan and Chen, Yiming and Feng, Xiaoduan and Cao, Jian and Wu, Yuxiong and Wang, Bin},
  journal={arXiv preprint arXiv:2508.13632},
  year={2025}
}
