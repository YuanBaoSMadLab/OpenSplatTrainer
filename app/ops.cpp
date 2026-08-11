// 自定义算子注册：torchvision::deform_conv2d（等价实现，纯 LibTorch 基础算子）。
// BiRefNet 导出为 TorchScript 后包含 torchvision::deform_conv2d 节点，
// C++ 端加载 rmbg.pt 前必须注册该算子。
// 算法与 torchvision deform_conv2d_kernel.cpp 完全一致：
//   1) offset 交错布局 [2*m]=h 偏移、[2*m+1]=w 偏移
//   2) 双线性插值，越界采样返回 0
//   3) y = out_y*stride_h - pad_h + i*dilation_h + offset_h
// 已验证（scripts/verify_deform.py）：与 torchvision 误差 < 3e-6。
#include <torch/script.h>
#include <torch/library.h>

namespace {

// 对 4 个角点分别 gather 并应用边界 mask
// C 分块处理：避免 idx4 展开到全 C 维（256x256、K=49 时长达 1.6GB）
torch::Tensor gather_clamped(
    const torch::Tensor& inp,          // [N,C,H,W]
    const torch::Tensor& hh,           // [N,K,Ho,Wo] long
    const torch::Tensor& ww,           // [N,K,Ho,Wo] long
    const torch::Tensor& valid,        // [N,K,Ho,Wo] bool
    int64_t c_chunk = 16) {
    int64_t N = inp.size(0), C = inp.size(1), H = inp.size(2), W = inp.size(3);
    int64_t K = hh.size(1), Ho = hh.size(2), Wo = hh.size(3);

    auto hc = hh.clamp(0, H - 1);
    auto wc = ww.clamp(0, W - 1);
    auto idx = (hc * W + wc).reshape({N, K * Ho * Wo});  // [N,K*HoWo]
    auto flat = inp.reshape({N, C, H * W});

    std::vector<torch::Tensor> parts;
    parts.reserve((C + c_chunk - 1) / c_chunk);
    for (int64_t c0 = 0; c0 < C; c0 += c_chunk) {
        int64_t c1 = std::min(C, c0 + c_chunk);
        auto idx4 = idx.unsqueeze(1).expand({N, c1 - c0, K * Ho * Wo}).contiguous();
        auto vals = flat.narrow(1, c0, c1 - c0).gather(2, idx4)
                        .reshape({N, c1 - c0, K, Ho, Wo});
        parts.push_back(std::move(vals));
    }
    auto vals = torch::cat(parts, 1);  // [N,C,K,Ho,Wo]

    auto vmask = valid.unsqueeze(1).expand({N, C, K, Ho, Wo});
    return torch::where(vmask, vals, torch::zeros_like(vals));
}

torch::Tensor deform_conv2d_impl(
    const torch::Tensor& input, const torch::Tensor& weight,
    const torch::Tensor& offset, const torch::Tensor& mask,
    const torch::Tensor& bias,
    c10::SymInt stride_h, c10::SymInt stride_w,
    c10::SymInt pad_h, c10::SymInt pad_w,
    c10::SymInt dilation_h, c10::SymInt dilation_w,
    c10::SymInt groups, c10::SymInt offset_groups,
    bool use_mask) {
  TORCH_CHECK(offset_groups == groups, "offset_groups must equal groups");
  TORCH_CHECK(use_mask, "mask is required");

  int64_t sh = stride_h.expect_int(), sw = stride_w.expect_int();
  int64_t ph = pad_h.expect_int(), pw = pad_w.expect_int();
  int64_t dh = dilation_h.expect_int(), dw = dilation_w.expect_int();

  // 内部统一用 float32 计算保证精度，返回时转回输入 dtype（fp16 模型内
  // 后续 BatchNorm 要求 input 与 weight dtype 一致，否则报类型不匹配）。
  auto out_dtype = input.scalar_type();
  auto x = input.contiguous().to(torch::kFloat32);
  auto wgt = weight.contiguous().to(torch::kFloat32);
  auto off = offset.contiguous().to(torch::kFloat32);
  auto msk = mask.contiguous().to(torch::kFloat32);
  auto bs = bias.defined() ? bias.contiguous().to(torch::kFloat32)
                           : torch::zeros({wgt.size(0)}, wgt.options());

  int64_t N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
  int64_t Cout = wgt.size(0), kh = wgt.size(2), kw = wgt.size(3);
  int64_t Ho = (H + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
  int64_t Wo = (W + 2 * pw - dw * (kw - 1) - 1) / sw + 1;
  int64_t K = kh * kw;

  auto opt = x.options();
  auto oy = torch::arange(Ho, opt).view({1, 1, Ho, 1});
  auto ox = torch::arange(Wo, opt).view({1, 1, 1, Wo});
  auto p = torch::arange(K, opt);
  auto kk_v = (p / kw).view({1, -1, 1, 1});
  auto ll_v = (p % kw).view({1, -1, 1, 1});

  // base 坐标 [1,K,Ho,Wo]
  auto base_y = oy * sh + kk_v * dh - ph;
  auto base_x = ox * sw + ll_v * dw - pw;

  // offset 交错布局
  auto off_h = off.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, torch::indexing::None, 2)});  // [N,K,Ho,Wo]
  auto off_w = off.index({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None, 2)});

  auto py = base_y + off_h;
  auto px = base_x + off_w;

  // 双线性插值
  auto y0 = torch::floor(py);
  auto x0 = torch::floor(px);
  auto ly = py - y0;
  auto lx = px - x0;
  auto hy = 1 - ly;
  auto hx = 1 - lx;

  auto y0l = y0.to(torch::kLong);
  auto x0l = x0.to(torch::kLong);
  auto y1l = y0l + 1;
  auto x1l = x0l + 1;

  auto in_bounds = [&](const torch::Tensor& yy, const torch::Tensor& xx) {
    return (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W);
  };

  auto w00 = (hy * hx).unsqueeze(1);
  auto w01 = (hy * lx).unsqueeze(1);
  auto w10 = (ly * hx).unsqueeze(1);
  auto w11 = (ly * lx).unsqueeze(1);

  // 分步累加：每个角点 gather 完即释放，避免 4 个 [N,C,K,Ho,Wo] 同时驻留
  auto sampled = torch::zeros({N, C, K, Ho, Wo}, x.options());
  {
    auto v00 = gather_clamped(x, y0l, x0l, in_bounds(y0l, x0l));
    sampled.addcmul_(w00.expand_as(sampled), v00);
  }
  {
    auto v01 = gather_clamped(x, y0l, x1l, in_bounds(y0l, x1l));
    sampled.addcmul_(w01.expand_as(sampled), v01);
  }
  {
    auto v10 = gather_clamped(x, y1l, x0l, in_bounds(y1l, x0l));
    sampled.addcmul_(w10.expand_as(sampled), v10);
  }
  {
    auto v11 = gather_clamped(x, y1l, x1l, in_bounds(y1l, x1l));
    sampled.addcmul_(w11.expand_as(sampled), v11);
  }
  sampled *= msk.unsqueeze(1);

  // 卷积：把 [N,C,K,Ho,Wo] × [Cout,C,K] 化为批量矩阵乘，内存低于 einsum
  auto wflat2 = wgt.reshape({Cout, C * K});                              // [Cout, C*K]
  auto s2 = sampled.view({N, C * K, Ho * Wo});                           // [N, C*K, HoWo]
  auto out = torch::bmm(s2.transpose(1, 2), wflat2.t().unsqueeze(0).expand({N, C * K, Cout}))
                 .transpose(1, 2)
                 .view({N, Cout, Ho, Wo});
  out = out + bs.view({1, Cout, 1, 1});
  return out.to(out_dtype);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def(
      "deform_conv2d(Tensor input, Tensor weight, Tensor offset, Tensor mask, "
      "Tensor bias, SymInt stride_h, SymInt stride_w, SymInt pad_h, SymInt pad_w, "
      "SymInt dilation_h, SymInt dilation_w, SymInt groups, SymInt offset_groups, "
      "bool use_mask) -> Tensor",
      deform_conv2d_impl);
}
