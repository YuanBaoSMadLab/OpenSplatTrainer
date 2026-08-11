"""生成 Python 参考输出，用于对比 C++ 结果是否一致。"""
import sys, os
sys.path.insert(0, os.path.abspath("TripoSplat"))
import torch
from triposplat import TripoSplatPipeline

pipe = TripoSplatPipeline(
    ckpt_path="TripoSplat/ckpts/diffusion_models/triposplat_fp16.safetensors",
    decoder_path="TripoSplat/ckpts/vae/triposplat_vae_decoder_fp16.safetensors",
    dinov3_path="TripoSplat/ckpts/clip_vision/dino_v3_vit_h.safetensors",
    flux2_vae_encoder_path="TripoSplat/ckpts/vae/flux2-vae.safetensors",
    rmbg_path="TripoSplat/ckpts/background_removal/birefnet.safetensors",
    device="cuda",
)
g, prepared = pipe.run(
    "TripoSplat/static/example_inputs/building_stone_house.webp",
    seed=42, steps=5, guidance_scale=3.0, shift=3.0, num_gaussians=32768,
    show_progress=False,
)
g.save_ply("test_data/ref.ply")
print("ref saved, gaussians:", g.get_xyz.shape[0])
