"""TripoSplat → TorchScript 一次性导出脚本。
运行时无 Python：本脚本只在开发期执行一次，把 6 个模型导出为 .pt，
C++ 应用用 LibTorch 加载。

用法：
  python export_models.py --triposplat-dir ../TripoSplat --out-dir ../models [--device cuda]
"""
import argparse
import os
import sys

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--triposplat-dir", required=True, help="TripoSplat 项目目录")
    p.add_argument("--out-dir", required=True, help="导出 .pt 的输出目录")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--skip-rmbg", action="store_true", help="跳过 BiRefNet 导出（调试用）")
    return p.parse_args()


def main():
    args = parse_args()
    ts_dir = os.path.abspath(args.triposplat_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    sys.path.insert(0, ts_dir)

    import model as M
    from triposplat import FLOW_MODEL_ARGS, OCTREE_DECODER_ARGS, GS_DECODER_ARGS

    device = torch.device(args.device)
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    torch.manual_seed(0)

    ckpt = os.path.join(ts_dir, "ckpts")
    paths = {
        "rmbg": os.path.join(ckpt, "background_removal", "birefnet.safetensors"),
        "dinov3": os.path.join(ckpt, "clip_vision", "dino_v3_vit_h.safetensors"),
        # 与 run_example.py 一致：Flux2VAEEncoder 从完整 Flux2-VAE ckpt 加载 encoder 部分
        "vae_encoder": os.path.join(ckpt, "vae", "flux2-vae.safetensors"),
        "flow": os.path.join(ckpt, "diffusion_models", "triposplat_fp16.safetensors"),
        "decoder": os.path.join(ckpt, "vae", "triposplat_vae_decoder_fp16.safetensors"),
    }

    # ---------- 1. BiRefNet 抠图 ----------
    if not args.skip_rmbg:
        print("[export] loading rmbg ...")
        rmbg = M.BiRefNet()
        rmbg.load_safetensors(paths["rmbg"])
        rmbg.to(device=device, dtype=dtype).eval()
        x = torch.randn(1, 3, 1024, 1024, device=device, dtype=dtype)
        sm = torch.jit.trace(rmbg, (x,), strict=False)
        sm.save(os.path.join(out_dir, "rmbg.pt"))
        print("  -> rmbg.pt saved")

    # ---------- 2. DINOv3 ----------
    print("[export] loading dinov3 ...")
    dinov3 = M.DinoV3ViT()
    dinov3.load_safetensors(paths["dinov3"])
    dinov3.to(device=device, dtype=dtype).eval()
    x = torch.randn(1, 3, 1024, 1024, device=device, dtype=dtype)
    sm = torch.jit.trace(dinov3, (x,), strict=False)
    sm.save(os.path.join(out_dir, "dinov3.pt"))
    print("  -> dinov3.pt saved")

    # ---------- 3. Flux2VAEEncoder ----------
    print("[export] loading vae_encoder ...")
    vae = M.Flux2VAEEncoder()
    vae.load_safetensors(paths["vae_encoder"])
    vae.to(device=device, dtype=dtype).eval()

    class VaeWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            # 确定性编码，与 pipeline encode_image 保持一致
            return self.m.encode(x, deterministic=True, generator=None)

    vw = VaeWrapper(vae)
    x = torch.randn(1, 3, 1024, 1024, device=device, dtype=dtype)
    sm = torch.jit.trace(vw, (x,), strict=False)
    sm.save(os.path.join(out_dir, "vae_encoder.pt"))
    print("  -> vae_encoder.pt saved")

    # ---------- 4. Flow model ----------
    print("[export] loading flow_model ...")
    fm = M.LatentSeqMMFlowModel(**FLOW_MODEL_ARGS)
    fm.load_safetensors(paths["flow"])
    fm.to(device=device, dtype=dtype).eval()

    class FlowWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, latent, camera, t, feature1, feature2):
            x_t = {"latent": latent, "camera": camera}
            cond = {"feature1": feature1, "feature2": feature2}
            out = self.m(x_t, t, cond)
            return out["latent"], out["camera"]

    fw = FlowWrapper(fm)
    latent = torch.randn(1, 8192, 16, device=device, dtype=dtype)
    camera = torch.randn(1, 1, 5, device=device, dtype=dtype)
    t = torch.tensor([500.0], device=device)
    f1 = torch.randn(1, 4101, 1280, device=device, dtype=dtype)
    f2 = torch.randn(1, 4101, 128, device=device, dtype=dtype)
    # warmup：让 pos_pe 等内部缓存稳定在目标设备，避免 trace sanity check 值不一致
    with torch.no_grad():
        fw(latent, camera, t, f1, f2)
    sm = torch.jit.trace(fw, (latent, camera, t, f1, f2), strict=False, check_trace=False)
    sm.save(os.path.join(out_dir, "flow_model.pt"))
    print("  -> flow_model.pt saved")

    # ---------- 5. Octree + GS decoder ----------
    print("[export] loading decoder ...")
    dec = M.OctreeGaussianDecoder(OCTREE_DECODER_ARGS, GS_DECODER_ARGS)
    dec.load_safetensors(paths["decoder"])
    dec.to(device=device, dtype=dtype).eval()

    octree = dec.octree

    class OctreeWrapper(torch.nn.Module):
        """返回 logits（而非 dict），便于 C++ 侧直接取 tensor。
        与 OctreeProbabilityFixedlenDecoder.sample 的调用签名一致：
        forward(parent_coords_norm, res_tensor, cond, num_tensor)。"""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x, l, cond, l2):
            return self.m(x, l, cond, l2)["logits"]

    ow = OctreeWrapper(octree)
    x = torch.randn(1, 8192, 3, device=device, dtype=dtype)
    l = torch.full((1,), 256, dtype=torch.long, device=device)
    cond = torch.randn(1, 8192, 16, device=device, dtype=dtype)
    l2 = torch.full((1,), 8192, dtype=torch.long, device=device)
    with torch.no_grad():
        ow(x, l, cond, l2)
    sm = torch.jit.trace(ow, (x, l, cond, l2), strict=False, check_trace=False)
    sm.save(os.path.join(out_dir, "octree.pt"))
    print("  -> octree.pt saved")

    gs = dec.gs

    class GsWrapper(torch.nn.Module):
        """完整 GS 解码：features → offset → 组装 → 激活。
        返回 (xyz_norm, features_dc, opacity, scaling, rotation)，
        C++ 端只做 aabb/变换与 PLY 写出。"""

        def __init__(self, m):
            super().__init__()
            self.m = m
            self.ng = m.rep_config['num_gaussians']
            self.use_learned_offset_scale = m.use_learned_offset_scale
            self.perturb_offset = m.rep_config['perturb_offset']
            self.perturbe_size = m.rep_config['perturbe_size']
            self.lr = m.rep_config['lr']
            self.offset_scale = m.rep_config['offset_scale']
            self.min_kernel = m.rep_config['filter_kernel_size_3d']
            self.scaling_bias = m.rep_config['scaling_bias']
            self.opacity_bias = m.rep_config['opacity_bias']
            # buffers
            self.register_buffer("points_offset_perturbation", m.points_offset_perturbation)
            self.register_buffer("base_offset_scale", m.base_offset_scale)
            # Gaussian 偏置采用逆激活值（与原版 Gaussian 类一致）
            #  softplus 逆：log(exp(x)-1)；sigmoid 逆：log(x/(1-x))
            self.register_buffer("scale_bias",
                                 torch.log(torch.exp(torch.tensor(self.scaling_bias, device="cuda")) - 1))
            self.register_buffer("opacity_bias_val",
                                 torch.log(torch.tensor(self.opacity_bias, device="cuda") / (1 - self.opacity_bias)))
            # layout
            self.layout_names = []
            self.layout_ranges = []
            start = 0
            for k, v in m.layout.items():
                self.layout_names.append(k)
                self.layout_ranges.append((v['range'][0], v['range'][1]))
                start = v['range'][1]
            self.register_buffer("rots_bias", torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda"))

        def _get_offset(self, h):
            B = h.shape[0]
            r_scale = self.layout_ranges[self.layout_names.index('_offset_scale')]
            _offset_scale = torch.nn.functional.softplus(
                h[:, :, r_scale[0]:r_scale[1]].reshape(B, -1, self.ng, 1)
                + self.base_offset_scale)
            r_xyz = self.layout_ranges[self.layout_names.index('_xyz')]
            offset = h[:, :, r_xyz[0]:r_xyz[1]].reshape(B, -1, self.ng, 3)
            offset = offset * self.lr['_xyz']
            if self.perturb_offset:
                offset = offset + self.points_offset_perturbation
            offset = torch.tanh(offset) * 0.5 * self.perturbe_size
            offset = offset * (_offset_scale if self.use_learned_offset_scale
                               else self.offset_scale)
            return offset

        def forward(self, points, cond):
            h = self.m(x={"points": points}, cond=cond)["features"]
            offset = self._get_offset(h)
            xyz_norm = (offset + points[:, :, None, :]).flatten(1, 2)

            def slice_attr(name, shape):
                r = self.layout_ranges[self.layout_names.index(name)]
                return h[:, :, r[0]:r[1]].reshape(h.shape[0], -1, *shape)

            features_dc = slice_attr('_features_dc', (1, 3))
            features_dc = features_dc * self.lr['_features_dc']
            opacity = slice_attr('_opacity', (1,)) * self.lr['_opacity']
            opacity = torch.sigmoid(opacity + self.opacity_bias_val)
            scaling = slice_attr('_scaling', (3,)) * self.lr['_scaling']
            scaling = torch.sqrt(torch.square(torch.nn.functional.softplus(
                scaling + self.scale_bias)) + self.min_kernel ** 2)
            rotation = slice_attr('_rotation', (4,)) * self.lr['_rotation']
            rotation = rotation + self.rots_bias
            return xyz_norm, features_dc, opacity, scaling, rotation

    gw = GsWrapper(gs)
    pts = torch.randn(1, 8192, 3, device=device, dtype=dtype)
    with torch.no_grad():
        gw(pts, cond)
    sm = torch.jit.trace(gw, (pts, cond), strict=False, check_trace=False)
    sm.save(os.path.join(out_dir, "gs_decoder.pt"))
    print("  -> gs_decoder.pt saved")

    print("\n导出完成，模型目录:", out_dir)
    for f in sorted(os.listdir(out_dir)):
        if f.endswith(".pt"):
            print(f"  {f}  ({os.path.getsize(os.path.join(out_dir, f)) // (1024*1024)} MB)")


if __name__ == "__main__":
    main()
