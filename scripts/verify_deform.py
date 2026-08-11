"""验证 deform_conv2d 矢量化解法，复刻 torchvision 语义。
关键点（来自 torchvision deform_conv2d_kernel.cpp）：
1. offset 通道布局：[2*mask_idx] 是 h 偏移，[2*mask_idx+1] 是 w 偏移（交错）
2. 越界采样返回 0（h<=-1 或 h>=height 或 w<=-1 或 w>=width）
3. 坐标：y = out_y*stride_h - pad_h + i*dilation_h + offset_h
"""
import torch
import torchvision

torch.manual_seed(0)
N, C, H, W = 2, 4, 16, 16
Cout, kh, kw = 8, 3, 3
x = torch.randn(N, C, H, W)
w = torch.randn(Cout, C, kh, kw)
b = torch.randn(Cout)
offset = torch.randn(N, 2 * kh * kw, H, W) * 2
mask = torch.sigmoid(torch.randn(N, kh * kw, H, W))

ref = torchvision.ops.deform_conv2d(x, offset, w, b, stride=(1, 1), padding=(1, 1),
                                    dilation=(1, 1), mask=mask)


def bilinear_interp(inp, h, w_coord, H, W):
    """inp: [N,C,H,W]，h/w_coord: [N,9,Ho,Wo]，返回采样值，越界为 0。"""
    N, C = inp.shape[0], inp.shape[1]
    out = torch.zeros(N, C, *h.shape[1:], device=inp.device, dtype=inp.dtype)
    h_low = torch.floor(h).long()
    w_low = torch.floor(w_coord).long()
    h_high = h_low + 1
    w_high = w_low + 1
    lh = h - h_low.float()
    lw = w_coord - w_low.float()
    hh = 1 - lh
    hw = 1 - lw

    v1 = torch.zeros_like(out)
    m = (h_low >= 0) & (w_low >= 0)
    m &= h_low < H
    m &= w_low < W
    if m.any():
        idx = h_low.clamp(0, H - 1) * W + w_low.clamp(0, W - 1)  # [N,9,Ho,Wo]
        idx = idx.unsqueeze(1).expand(N, C, -1, -1, -1)
        flat = inp.reshape(N, C, H * W)
        vals = flat.gather(2, idx.reshape(N, C, -1)).reshape(N, C, *h.shape[1:])
        v1 = torch.where(m.unsqueeze(1), vals, torch.zeros_like(vals))

    v2 = torch.zeros_like(out)
    m = (h_low >= 0) & (w_high <= W - 1) & (h_low < H) & (w_high >= 0)
    if m.any():
        idx = h_low.clamp(0, H - 1) * W + w_high.clamp(0, W - 1)
        idx = idx.unsqueeze(1).expand(N, C, -1, -1, -1)
        flat = inp.reshape(N, C, H * W)
        vals = flat.gather(2, idx.reshape(N, C, -1)).reshape(N, C, *h.shape[1:])
        v2 = torch.where(m.unsqueeze(1), vals, torch.zeros_like(vals))

    v3 = torch.zeros_like(out)
    m = (h_high <= H - 1) & (w_low >= 0) & (h_high >= 0) & (w_low < W)
    if m.any():
        idx = h_high.clamp(0, H - 1) * W + w_low.clamp(0, W - 1)
        idx = idx.unsqueeze(1).expand(N, C, -1, -1, -1)
        flat = inp.reshape(N, C, H * W)
        vals = flat.gather(2, idx.reshape(N, C, -1)).reshape(N, C, *h.shape[1:])
        v3 = torch.where(m.unsqueeze(1), vals, torch.zeros_like(vals))

    v4 = torch.zeros_like(out)
    m = (h_high <= H - 1) & (w_high <= W - 1) & (h_high >= 0) & (w_high >= 0)
    if m.any():
        idx = h_high.clamp(0, H - 1) * W + w_high.clamp(0, W - 1)
        idx = idx.unsqueeze(1).expand(N, C, -1, -1, -1)
        flat = inp.reshape(N, C, H * W)
        vals = flat.gather(2, idx.reshape(N, C, -1)).reshape(N, C, *h.shape[1:])
        v4 = torch.where(m.unsqueeze(1), vals, torch.zeros_like(vals))

    w1 = (hh * hw).unsqueeze(1)
    w2 = (hh * lw).unsqueeze(1)
    w3 = (lh * hw).unsqueeze(1)
    w4 = (lh * lw).unsqueeze(1)
    return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4


def deform_conv_vectorized(x, offset, weight, bias, stride, padding, dilation, mask):
    N, C, H, W = x.shape
    Cout, _, kh, kw = weight.shape
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    Ho = (H + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    Wo = (W + 2 * pw - dw * (kw - 1) - 1) // sw + 1

    oy = torch.arange(Ho, device=x.device).float()
    ox = torch.arange(Wo, device=x.device).float()
    p = torch.arange(kh * kw, device=x.device)
    kk_v = (p // kw).view(1, -1, 1, 1).float()
    ll_v = (p % kw).view(1, -1, 1, 1).float()

    base_y = oy.view(1, 1, Ho, 1) * sh + kk_v * dh - ph  # [1,9,Ho,1]
    base_x = ox.view(1, 1, 1, Wo) * sw + ll_v * dw - pw  # [1,9,1,Wo]

    # offset 交错布局：[2*m] = h, [2*m+1] = w
    off_h = offset[:, 0::2]  # [N,9,Ho,Wo]
    off_w = offset[:, 1::2]
    py = base_y + off_h
    px = base_x + off_w

    sampled = bilinear_interp(x, py, px, H, W)  # [N,C,9,Ho,Wo]
    msk = mask.unsqueeze(1)  # [N,1,9,Ho,Wo]
    sampled = sampled * msk

    wflat = weight.reshape(Cout, C, kh * kw)
    out = torch.einsum('ncpqw,ocp->noqw', sampled, wflat)
    out += bias.view(1, Cout, 1, 1)
    return out


out = deform_conv_vectorized(x, offset, w, b, (1, 1), (1, 1), (1, 1), mask)
print('vectorized vs torchvision max diff:', (ref - out).abs().max().item())
