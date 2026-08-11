// TripoSplat 推理内核实现（LibTorch C++）。
#include "infer.h"
#include "image_utils.h"
#include "ply_writer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>
#include <torch/cuda.h>
#include <torch/nn/functional.h>
#include <c10/cuda/CUDACachingAllocator.h>

namespace ost {

using torch::Tensor;
using torch::jit::Module;

// 调试：分阶段推理时记录各阶段显存（写入工作目录 vram_trace.txt）
static void vram_trace(const char* tag) {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return;
    FILE* f = nullptr;
    fopen_s(&f, "vram_trace.txt", "a");
    if (f) {
        fprintf(f, "[%s] free=%.2fGB total=%.2fGB\n", tag,
                free_b / 1073741824.0, total_b / 1073741824.0);
        fclose(f);
    }
}

namespace {

struct Models {
    Module rmbg;
    Module dinov3;
    Module vae_encoder;
    Module flow_model;
    Module octree;
    Module gs_decoder;
};

const int kCanvasSize = 1024;
const int kFlowTokens = 8192;
const int kFlowInChannels = 16;
const int kCamChannels = 5;
const int kCondChannels = 1280;
const int kCond2Channels = 128;

torch::Device g_device = torch::kCUDA;

inline torch::TensorOptions dev_opts(torch::Dtype dtype = torch::kFloat32) {
    return torch::TensorOptions().device(g_device).dtype(dtype);
}

Tensor to_device(const Tensor& t, torch::Dtype dtype = torch::kFloat16) {
    return t.to(g_device, dtype);
}

// ---------------------------------------------------------------------------
// 预处理：BiRefNet 抠图 → 腐蚀 → bbox 裁剪 → canvas_size 黑底合成
// 返回 [1,3,canvas_size,canvas_size] float32（0~1，黑底 RGB）
// rmbg_res：抠图模型输入分辨率（低显存时降到 512/768）
// canvas_size：输入分辨率（min 边缩放目标，512/768/1024/1536）
// ---------------------------------------------------------------------------
Tensor preprocess(const std::string& image_path, Module& rmbg, int erode_radius,
                  int rmbg_res, int canvas_size) {
    DecodedImage img;
    std::string err;
    if (!decode_image_wic(image_path, img, err)) {
        throw std::runtime_error(err);
    }

    // 1. resize 使 min 边 = canvas_size（保持宽高比）
    int w = img.width, h = img.height;
    float s = (float)canvas_size / std::min(w, h);
    int nw = std::max(1, (int)std::lround(w * s));
    int nh = std::max(1, (int)std::lround(h * s));
    std::vector<uint8_t> resized((size_t)nw * nh * 4);
    resize_lanczos(img.rgba.data(), w, h, resized.data(), nw, nh);

    // 2. 判断是否有真实 alpha
    bool has_real_alpha = false;
    if (img.rgba.size() >= 4) {
        for (size_t i = 3; i < img.rgba.size(); i += 4) {
            if (img.rgba[i] < 255) { has_real_alpha = true; break; }
        }
    }

    std::vector<uint8_t> rgba = resized;  // [nw,nh,4]

    if (!has_real_alpha) {
        // BiRefNet 抠图：RGB → 归一化 → 模型 → alpha 回缩放
        Tensor rgb_t = torch::zeros({1, 3, nh, nw}, torch::kFloat32);
        {
            auto acc = rgb_t.accessor<float, 4>();
            for (int y = 0; y < nh; ++y) {
                for (int x = 0; x < nw; ++x) {
                    const uint8_t* p = rgba.data() + ((size_t)y * nw + x) * 4;
                    acc[0][0][y][x] = p[0] / 255.0f;
                    acc[0][1][y][x] = p[1] / 255.0f;
                    acc[0][2][y][x] = p[2] / 255.0f;
                }
            }
        }
        // resize 到抠图分辨率（低显存时降级，alpha 质量略降但省显存）
        int mr = rmbg_res > 0 ? rmbg_res : canvas_size;
        Tensor t1024 = torch::nn::functional::interpolate(
            rgb_t,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{mr, mr})
                .mode(torch::kBilinear)
                .align_corners(true));
        Tensor mean = torch::tensor({0.485f, 0.456f, 0.406f}, torch::kFloat32).view({1, 3, 1, 1});
        Tensor std = torch::tensor({0.229f, 0.224f, 0.225f}, torch::kFloat32).view({1, 3, 1, 1});
        Tensor normed = (t1024 - mean) / std;

        Tensor alpha1024 = rmbg.forward({to_device(normed, torch::kFloat16)}).toTensor();
        alpha1024 = alpha1024[0][0];  // [1024,1024]

        // 回缩放 alpha 到原图尺寸
        Tensor alpha = torch::nn::functional::interpolate(
            alpha1024.unsqueeze(0).unsqueeze(0).to(torch::kFloat32),
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{nh, nw})
                .mode(torch::kBilinear)
                .align_corners(true))[0][0].clamp(0, 1);
        // accessor 只能访问 CPU 张量，先搬到 CPU
        auto alpha_cpu = alpha.to(torch::kCPU);
        auto alpha_acc = alpha_cpu.accessor<float, 2>();
        for (int y = 0; y < nh; ++y) {
            for (int x = 0; x < nw; ++x) {
                rgba[(size_t)y * nw * 4 + x * 4 + 3] = (uint8_t)(alpha_acc[y][x] * 255.0f);
            }
        }
    }

    // 3. 腐蚀 alpha
    std::vector<uint8_t> alpha(nw * nh);
    for (int y = 0; y < nh; ++y)
        for (int x = 0; x < nw; ++x)
            alpha[(size_t)y * nw + x] = rgba[(size_t)y * nw * 4 + x * 4 + 3];
    if (erode_radius > 0) {
        min_filter(alpha.data(), nw, nh, erode_radius);
        for (int y = 0; y < nh; ++y)
            for (int x = 0; x < nw; ++x)
                rgba[(size_t)y * nw * 4 + x * 4 + 3] = alpha[(size_t)y * nw + x];
    }

    // 4. bbox 裁剪（外扩 1.2 倍）→ canvas_size 黑底合成
    int x0, y0, x1, y1;
    alpha_bbox(alpha.data(), nw, nh, x0, y0, x1, y1);
    int cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    int half = (int)std::lround(std::max(x1 - x0, y1 - y0) / 2.0f * 1.2f);
    int bx0 = std::clamp(cx - half, 0, nw - 1);
    int by0 = std::clamp(cy - half, 0, nh - 1);
    int bx1 = std::clamp(cx + half, 0, nw - 1);
    int by1 = std::clamp(cy + half, 0, nh - 1);
    if (bx1 <= bx0) { bx0 = 0; bx1 = nw - 1; }
    if (by1 <= by0) { by0 = 0; by1 = nh - 1; }

    std::vector<uint8_t> canvas((size_t)canvas_size * canvas_size * 4, 0);
    crop_to_black(rgba.data(), nw, nh, bx0, by0, bx1, by1,
                  canvas.data(), canvas_size, canvas_size);

    // 5. 转 tensor [1,3,canvas_size,canvas_size]
    Tensor out = torch::zeros({1, 3, canvas_size, canvas_size}, torch::kFloat32);
    {
        auto acc = out.accessor<float, 4>();
        for (int y = 0; y < canvas_size; ++y) {
            for (int x = 0; x < canvas_size; ++x) {
                const uint8_t* p = canvas.data() + ((size_t)y * canvas_size + x) * 4;
                acc[0][0][y][x] = p[0] / 255.0f;
                acc[0][1][y][x] = p[1] / 255.0f;
                acc[0][2][y][x] = p[2] / 255.0f;
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// 编码：DINOv3 + Flux2VAE
// 返回 {feature1: [1,4101,1280] float32, feature2: [1,4101,128] float32}
// ---------------------------------------------------------------------------
struct Encoded {
    Tensor feature1;
    Tensor feature2;
};

Encoded encode(const Tensor& prepared, Models& m) {
    // DINOv3：ImageNet 归一化
    Tensor mean = torch::tensor({0.485f, 0.456f, 0.406f}, torch::kFloat32).view({1, 3, 1, 1});
    Tensor std = torch::tensor({0.229f, 0.224f, 0.225f}, torch::kFloat32).view({1, 3, 1, 1});
    Tensor normed = (prepared - mean) / std;

    Tensor feat1 = m.dinov3.forward({to_device(normed, torch::kFloat16)}).toTensor();
    feat1 = feat1.to(torch::kFloat32);
    int64_t last_dim = feat1.size(-1);
    feat1 = torch::layer_norm(feat1, {last_dim});

    // Flux2VAE encode（确定性模式，与导出一致）
    Tensor x2 = prepared * 2 - 1;
    Tensor vae_feat = m.vae_encoder.forward({to_device(x2, torch::kFloat16)}).toTensor();

    // pad 5 zero tokens：zero_reg [1,5,C] + vae_feat
    int64_t C = vae_feat.size(-1);
    Tensor zero_reg = torch::zeros({vae_feat.size(0), 5, C},
                                   vae_feat.options().dtype(torch::kFloat16));
    Tensor feat2 = torch::cat({zero_reg, vae_feat}, 1).to(torch::kFloat32);

    return {feat1, feat2};
}

// ---------------------------------------------------------------------------
// Euler flow sampling with CFG
// 对应 FlowEulerCfgSampler.sample + _cfg_prediction
// ---------------------------------------------------------------------------
Tensor sample_latent(Models& m, const Tensor& feat1, const Tensor& feat2,
                     int steps, float guidance_scale, float shift,
                     const std::function<void(int cur, int total)>& on_step = {}) {
    // noise（使用全局默认 generator，已通过 manual_seed 固定）
    Tensor latent = torch::randn({1, kFlowTokens, kFlowInChannels},
                                 dev_opts(torch::kFloat16));
    Tensor camera = torch::randn({1, 1, kCamChannels},
                                 dev_opts(torch::kFloat16));

    Tensor cond1 = feat1.to(torch::kFloat16);
    Tensor cond2 = feat2.to(torch::kFloat16);
    Tensor neg1 = torch::zeros_like(cond1);
    Tensor neg2 = torch::zeros_like(cond2);

    // t_seq = shift * linspace(1,0,steps+1) / (1 + (shift-1)*linspace)
    std::vector<double> lin(steps + 1);
    for (int i = 0; i <= steps; ++i) lin[i] = 1.0 - (double)i / steps;
    std::vector<float> t_seq(steps + 1);
    for (int i = 0; i <= steps; ++i) {
        t_seq[i] = (float)(shift * lin[i] / (1 + (shift - 1) * lin[i]));
    }

    for (int i = 0; i < steps; ++i) {
        float t = t_seq[i], t_prev = t_seq[i + 1];
        float dt = t - t_prev;

        // cond pass
        Tensor t_scaled = torch::full({1}, 1000.0f * t, dev_opts(torch::kFloat32));
        auto out_c = m.flow_model.forward({latent, camera, t_scaled, cond1, cond2}).toTuple();
        Tensor pred_latent = out_c->elements()[0].toTensor();
        Tensor pred_cam = out_c->elements()[1].toTensor();

        if (guidance_scale > 1.0f) {
            Tensor t_neg = torch::full({1}, 1000.0f * t, dev_opts(torch::kFloat32));
            auto out_n = m.flow_model.forward({latent, camera, t_neg, neg1, neg2}).toTuple();
            Tensor neg_latent = out_n->elements()[0].toTensor();
            Tensor neg_cam = out_n->elements()[1].toTensor();
            pred_latent = guidance_scale * pred_latent - (guidance_scale - 1) * neg_latent;
            pred_cam = guidance_scale * pred_cam - (guidance_scale - 1) * neg_cam;
        }

        latent = latent - pred_latent * dt;
        camera = camera - pred_cam * dt;

        if (on_step) on_step(i, steps);
    }
    return latent;  // [1,8192,16] fp16
}

// ---------------------------------------------------------------------------
// systematic 采样（复刻 Python 端 sample_probs）
// probs: [B,P] float32, counts: [B] long → out [B,P] long
// ---------------------------------------------------------------------------
Tensor sample_probs(const Tensor& probs, const Tensor& counts) {
    int64_t B = counts.numel();
    int64_t P = probs.size(-1);
    Tensor p = probs.reshape({B, P}).to(torch::kFloat32).clamp_min(0);
    Tensor row_sums = p.sum(1, true);
    p = p / row_sums.clamp_min(1);
    Tensor zero_mask = row_sums.eq(0);
    if (zero_mask.any().item<bool>()) {
        p = p.clone();
        p.masked_fill_(zero_mask.expand_as(p), 1.0 / (double)P);
    }

    Tensor counts_l = counts.reshape({B}).to(torch::kLong);
    Tensor out = torch::zeros({B, P}, dev_opts(torch::kLong));
    Tensor cdf = p.cumsum(1).clamp_max(1.0 - 1e-12);

    // unique 分组处理（与原版一致：对每个唯一计数分别采样）
    auto uniq = torch::_unique(counts_l, false, true);
    Tensor unique_n = std::get<0>(uniq);  // 值
    Tensor inv = std::get<1>(uniq);       // 逆映射

    int64_t n_unique = unique_n.size(0);
    for (int64_t i = 0; i < n_unique; ++i) {
        int64_t n = unique_n[i].item<int64_t>();
        if (n == 0) continue;
        Tensor rows = (inv == i).nonzero().squeeze(1);
        int64_t r = rows.size(0);
        if (r == 0) continue;

        Tensor U0 = torch::rand({r, 1}, dev_opts(torch::kFloat32)) / (float)n;
        Tensor grid = torch::arange(n, dev_opts(torch::kFloat32)).view({1, n}) / (float)n;
        Tensor us = (U0 + grid).clamp_max(1.0 - 1e-12);
        Tensor cdf_rows = cdf.index_select(0, rows);
        Tensor idx = torch::searchsorted(cdf_rows, us, false).clamp_max(P - 1);
        Tensor buf = torch::zeros({r, P}, dev_opts(torch::kFloat32));
        Tensor ones = torch::ones_like(idx, dev_opts(torch::kFloat32));
        buf.scatter_add_(1, idx, ones);
        out.index_copy_(0, rows, buf.to(torch::kLong));
    }
    return out;
}

// ---------------------------------------------------------------------------
// 八叉树采样（对应 OctreeProbabilityFixedlenDecoder.sample）
// cond: [1,8192,16], num_points → points [1,N,3]
// ---------------------------------------------------------------------------
Tensor octree_sample(Module& octree, const Tensor& cond, int64_t num_points, int level) {
    int64_t B = cond.size(0);
    torch::Device dev = cond.device();

    Tensor child_offset = torch::zeros({8, 3}, dev_opts(torch::kLong));
    {
        int idx = 0;
        for (int k = 0; k <= 1; ++k)
            for (int j = 0; j <= 1; ++j)
                for (int i = 0; i <= 1; ++i) {
                    child_offset[idx][0] = i;
                    child_offset[idx][1] = j;
                    child_offset[idx][2] = k;
                    idx++;
                }
    }

    Tensor prev_coords_int = torch::zeros({B, 1, 3}, dev_opts(torch::kLong));
    Tensor prev_counts = torch::full({B, 1}, num_points, dev_opts(torch::kLong));
    Tensor prev_log_probs = torch::zeros({B, 1}, dev_opts(torch::kFloat32));
    Tensor batch_indices = torch::arange(B, dev_opts(torch::kLong)).unsqueeze(1);
    Tensor num_tensor = torch::full({B}, num_points, dev_opts(torch::kLong));

    for (int lv = 1; lv <= level; ++lv) {
        int64_t res_p = 1LL << (lv - 1);
        int64_t res = 1LL << lv;
        Tensor parent_coords_norm = (prev_coords_int.to(torch::kFloat32) + 0.5f) / (float)res_p;
        Tensor res_tensor = torch::full({B}, res, dev_opts(torch::kLong));

        // octree forward: (parent_coords_norm, res_tensor, cond, num_tensor) → logits
        Tensor logits = octree.forward({parent_coords_norm, res_tensor, cond, num_tensor}).toTensor();
        logits = logits / 1.0f;  // temperature=1.0
        Tensor pred_probs = torch::softmax(logits, -1);
        Tensor pred_log_probs = torch::log_softmax(logits, -1);

        Tensor sampled = sample_probs(pred_probs, prev_counts).reshape({B, -1});
        Tensor pred_log_probs_f = pred_log_probs.reshape({B, -1});
        Tensor prev_log_probs_exp = prev_log_probs.repeat_interleave(8, 1);

        Tensor child_coords_int =
            (prev_coords_int.unsqueeze(2) * 2 + child_offset.view({1, 1, 8, 3})).reshape({B, -1, 3});
        Tensor mask = sampled > 0;

        int64_t max_valid = mask.sum(1).max().item<int64_t>();
        Tensor scatter_idx = mask.cumsum(1) - 1;
        Tensor valid_scatter = scatter_idx.masked_select(mask);
        Tensor valid_batch = batch_indices.expand_as(mask).masked_select(mask);
        Tensor child_masked = child_coords_int.masked_select(mask.unsqueeze(-1).expand_as(child_coords_int))
                                  .reshape({-1, 3});
        Tensor sampled_masked = sampled.masked_select(mask);
        Tensor logp_masked = (prev_log_probs_exp + pred_log_probs_f).masked_select(mask);

        Tensor next_coords = torch::zeros({B, max_valid, 3}, dev_opts(torch::kLong));
        Tensor next_counts = torch::zeros({B, max_valid}, dev_opts(torch::kLong));
        Tensor next_logp = torch::zeros({B, max_valid}, dev_opts(torch::kFloat32));
        next_coords.index_put_({valid_batch, valid_scatter}, child_masked);
        next_counts.index_put_({valid_batch, valid_scatter}, sampled_masked);
        next_logp.index_put_({valid_batch, valid_scatter}, logp_masked);

        prev_coords_int = next_coords;
        prev_counts = next_counts;
        prev_log_probs = next_logp;
    }

    // 最终重复展开到 num_points
    Tensor counts_flat = prev_counts.reshape({-1});
    Tensor coords_flat = prev_coords_int.reshape({-1, 3});
    int64_t res = 1LL << level;
    Tensor final_coords = torch::repeat_interleave(coords_flat, counts_flat, 0).reshape({B, num_points, 3});
    Tensor rand_off = torch::rand_like(final_coords, dev_opts(torch::kFloat32));
    Tensor coords_norm = (final_coords.to(torch::kFloat32) + rand_off) / (float)res;
    return coords_norm;  // [1,N,3] float32
}

// ---------------------------------------------------------------------------
// 主流程
// ---------------------------------------------------------------------------
Module load_module(const std::string& dir, const std::string& name) {
    try {
        auto m = torch::jit::load(dir + "/" + name);
        // 只转设备、保持 dtype：导出时参数已是 fp16，且 int64 索引 buffer
        // （如 BiRefNet 的 relative_position_index）不能随 dtype 一起被转成 fp16
        m.to(g_device);
        m.eval();
        return m;
    } catch (const std::exception& e) {
        throw std::runtime_error("load " + name + " failed: " + e.what());
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// InferEngine：常驻推理引擎（模型加载一次，可反复 run；lazy 模式按需加载/释放）
// ---------------------------------------------------------------------------
struct InferEngine::Impl {
    Models m;
    torch::Device dev = torch::kCPU;
    std::string models_dir;
    bool lazy = false;
    bool loaded = false;
    // 各模型独立加载标志（低显存模式分阶段按需加载/释放）
    bool h_rmbg = false, h_dinov3 = false, h_vae = false,
         h_flow = false, h_octree = false, h_gs = false;

    // 按需加载单个模型（已加载则跳过）
    bool ensure(Module& slot, bool& flag, const std::string& name) {
        if (flag) return true;
        slot = load_module(models_dir, name);
        flag = true;
        return true;
    }

    // 释放单个模型（释放其 GPU 显存）
    void drop(Module& slot, bool& flag) {
        if (flag) {
            slot = Module();
            flag = false;
        }
    }

    void cache_clear() {
        if (dev == torch::kCUDA) {
            try {
                at::cuda::CUDACachingAllocator::emptyCache();
            } catch (...) {
            }
        }
    }

    // 高显存模式：一次性加载全部（常驻复用）
    void load_models() {
        ensure(m.rmbg, h_rmbg, "rmbg.pt");
        ensure(m.dinov3, h_dinov3, "dinov3.pt");
        ensure(m.vae_encoder, h_vae, "vae_encoder.pt");
        ensure(m.flow_model, h_flow, "flow_model.pt");
        ensure(m.octree, h_octree, "octree.pt");
        ensure(m.gs_decoder, h_gs, "gs_decoder.pt");
        loaded = true;
    }

    void release_models() {
        drop(m.rmbg, h_rmbg);
        drop(m.dinov3, h_dinov3);
        drop(m.vae_encoder, h_vae);
        drop(m.flow_model, h_flow);
        drop(m.octree, h_octree);
        drop(m.gs_decoder, h_gs);
        loaded = false;
        cache_clear();
    }
};

InferEngine::InferEngine(const std::string& models_dir, bool use_cuda, bool lazy)
    : impl_(new Impl) {
    if (use_cuda && torch::cuda::is_available()) {
        g_device = torch::kCUDA;
        impl_->dev = torch::kCUDA;
    } else {
        g_device = torch::kCPU;
        impl_->dev = torch::kCPU;
    }
    impl_->models_dir = models_dir;
    impl_->lazy = lazy;
    if (!lazy) {
        impl_->load_models();
    }
}

InferEngine::~InferEngine() = default;

bool InferEngine::cuda_available() {
    return torch::cuda::is_available();
}

int InferEngine::run_impl(const InferOptions& opt, InferResult& out, std::string& err,
                          int attempt) {
    try {
        // 与 Python 端 @torch.no_grad() 对应：抑制 autograd 图构建，避免显存爆炸
        torch::NoGradGuard no_grad;

        auto& I = *impl_;
        // 高显存模式：一次性加载全部常驻复用；
        // 低显存模式：逐阶段按需加载/释放，峰值 = 单阶段模型权重 + 中间量（8GB 卡可用）
        if (!I.lazy && !I.loaded) {
            I.load_models();
        }

        torch::manual_seed(opt.seed);
        if (I.dev == torch::kCUDA) {
            torch::cuda::manual_seed(opt.seed);
        }

        auto t0 = std::chrono::steady_clock::now();

        // 进度上报（阶段权重：抠图 0-12，编码 12-22，采样 22-90，解码 90-100）
        auto report = [&](int pct, const std::string& label) {
            if (opt.progress_cb) opt.progress_cb(pct, label);
        };

        // 阶段 1：抠图（仅 rmbg 常驻）
        if (I.lazy) vram_trace("st1 before rmbg");
        report(3, "读取图片…");
        I.ensure(I.m.rmbg, I.h_rmbg, "rmbg.pt");
        if (I.lazy) vram_trace("st1 rmbg loaded");
        Tensor prepared = preprocess(opt.input_image, I.m.rmbg, opt.erode_radius,
                                     opt.rmbg_res, opt.canvas_res);
        report(12, "抠图完成");
        if (I.lazy) {
            I.drop(I.m.rmbg, I.h_rmbg);
            I.cache_clear();
            vram_trace("st1 done (rmbg dropped)");
        }

        // 阶段 2：编码（dinov3 + vae）
        report(15, "提取图像特征…");
        if (I.lazy) vram_trace("st2 before dino+vae");
        I.ensure(I.m.dinov3, I.h_dinov3, "dinov3.pt");
        I.ensure(I.m.vae_encoder, I.h_vae, "vae_encoder.pt");
        if (I.lazy) vram_trace("st2 dino+vae loaded");
        Encoded enc = encode(prepared, I.m);
        report(22, "特征就绪");
        if (I.lazy) {
            I.drop(I.m.dinov3, I.h_dinov3);
            I.drop(I.m.vae_encoder, I.h_vae);
            I.cache_clear();
            vram_trace("st2 done (dino+vae dropped)");
        }

        // 阶段 3：flow 采样
        report(24, "采样生成中…");
        if (I.lazy) vram_trace("st3 before flow");
        I.ensure(I.m.flow_model, I.h_flow, "flow_model.pt");
        if (I.lazy) vram_trace("st3 flow loaded");
        Tensor latent = sample_latent(I.m, enc.feature1, enc.feature2,
                                      opt.steps, opt.guidance_scale, opt.shift,
                                      [&](int cur, int total) {
                                          int pct = 24 + (int)std::lround(66.0 * (cur + 1) / std::max(1, total));
                                          report(pct, "采样 " + std::to_string(cur + 1) + "/" + std::to_string(total));
                                      });
        report(90, "采样完成");
        if (I.lazy) {
            I.drop(I.m.flow_model, I.h_flow);
            I.cache_clear();
            vram_trace("st3 done (flow dropped)");
        }

        // 阶段 4：octree + GS 解码
        report(92, "解码高斯…");
        if (I.lazy) vram_trace("st4 before octree+gs");
        I.ensure(I.m.octree, I.h_octree, "octree.pt");
        I.ensure(I.m.gs_decoder, I.h_gs, "gs_decoder.pt");
        if (I.lazy) vram_trace("st4 octree+gs loaded");
        int64_t num_decoder_tokens = std::max<int64_t>(1, opt.num_gaussians / 32);
        Tensor points = octree_sample(I.m.octree, latent, num_decoder_tokens, 8);

        // GS 解码：输出 [1,N,...]，flatten 为 [N,...]
        auto gs_out = I.m.gs_decoder.forward({points.to(torch::kFloat16), latent}).toTuple();
        Tensor xyz_norm = gs_out->elements()[0].toTensor().reshape({-1, 3});
        Tensor features_dc = gs_out->elements()[1].toTensor().reshape({-1, 3});
        Tensor opacity = gs_out->elements()[2].toTensor().reshape({-1, 1});
        Tensor scaling = gs_out->elements()[3].toTensor().reshape({-1, 3});
        Tensor rotation = gs_out->elements()[4].toTensor().reshape({-1, 4});
        if (I.lazy) {
            I.drop(I.m.octree, I.h_octree);
            I.drop(I.m.gs_decoder, I.h_gs);
            I.cache_clear();
            vram_trace("st4 done (all models dropped)");
        }
        report(100, "生成完成");

        int64_t n = xyz_norm.size(0);
        auto t1 = std::chrono::steady_clock::now();

        out.num_gaussians = n;
        out.xyz = xyz_norm;
        out.features_dc = features_dc;
        out.opacity = opacity;
        out.scaling = scaling;
        out.rotation = rotation;
        out.seconds = std::chrono::duration<double>(t1 - t0).count();

        // 写出（可选）：失败不致命——模型已成功生成，仅文件未导出（记警告）
        if (!opt.output_ply.empty()) {
            if (!write_ply(opt.output_ply, xyz_norm, features_dc.unsqueeze(1),
                           opacity, scaling, rotation, err)) {
                err = "[导出警告] PLY 写入失败: " + err;
            }
        }
        if (!opt.output_splat.empty()) {
            std::string serr;
            if (!write_splat(opt.output_splat, xyz_norm, features_dc.unsqueeze(1),
                             opacity, scaling, rotation, serr)) {
                if (!err.empty()) err += "；";
                err += "[导出警告] SPLAT 写入失败: " + serr;
            }
        }
        if (impl_->lazy) impl_->release_models();
        return 0;
    } catch (const std::exception& e) {
        std::string w = e.what();
        // OOM 自动降档重试：高斯数减半 + 抠图分辨率降 25%
        if (attempt == 0 && impl_->dev == torch::kCUDA &&
            w.find("out of memory") != std::string::npos) {
            InferOptions o2 = opt;
            o2.num_gaussians = std::max(4096, opt.num_gaussians / 2);
            o2.rmbg_res = std::max(384, opt.rmbg_res * 3 / 4);
            impl_->release_models();
            return run_impl(o2, out, err, 1);
        }
        if (impl_->lazy) impl_->release_models();
        err = w;
        return -1;
    }
}

int InferEngine::run(const InferOptions& opt, InferResult& out, std::string& err) {
    return run_impl(opt, out, err, 0);
}

// 便捷接口：一次性推理（每次重新加载模型，供 CLI 使用）。
int run_inference(const InferOptions& opt, std::string& err) {
    try {
        InferResult out;
        InferEngine engine(opt.models_dir, opt.use_cuda);
        int rc = engine.run(opt, out, err);
        if (rc < 0) return rc;
        return (int)out.num_gaussians;
    } catch (const std::exception& e) {
        err = e.what();
        return -1;
    }
}

}  // namespace ost
