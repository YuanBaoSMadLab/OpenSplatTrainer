// TripoSplat 推理内核（LibTorch C++）。
// 对应 triposplat.py 的 TripoSplatPipeline.run 全流程，无 Python。
#pragma once

#include <functional>
#include <memory>
#include <string>

#include <torch/script.h>

namespace ost {

struct InferOptions {
    std::string models_dir;   // 导出的 .pt 目录
    std::string input_image;  // 输入图片
    std::string output_ply;   // 输出 PLY 路径（可空，表示不写）
    std::string output_splat; // 输出 SPLAT 路径（可空）
    std::string output_preview; // 预处理图输出（可空）
    int seed = 42;
    int steps = 20;
    float guidance_scale = 3.0f;
    float shift = 3.0f;
    int num_gaussians = 262144;
    int erode_radius = 1;
    bool use_cuda = true;
    int rmbg_res = 1024;     // BiRefNet 抠图分辨率（512/768/1024），低显存时降级
    int canvas_res = 1024;   // 输入分辨率：预处理将 min 边缩放至此并合成画布（512/768/1024/1536）
    bool lazy_models = false; // 低显存：run 前加载模型、跑完释放（显存峰值低，稍慢）
    // 进度回调（worker 线程调用）：pct 0~100，label 为阶段描述
    std::function<void(int pct, const std::string& label)> progress_cb;
};

// 推理结果（供 3D 查看器/导出使用）。张量保持在推理设备（GPU）。
struct InferResult {
    int64_t num_gaussians = 0;
    torch::Tensor xyz;         // [N,3] fp32
    torch::Tensor features_dc; // [N,3] SH0 系数（PLY 中即 f_dc 三通道）
    torch::Tensor opacity;     // [N,1] sigmoid 后 0~1
    torch::Tensor scaling;     // [N,3] 实际缩放（已激活）
    torch::Tensor rotation;    // [N,4] 单位四元数
    double seconds = 0.0;
};

// 推理引擎：默认构造时加载全部模型，可反复调用 run（lazy 模式按需加载/释放）。
class InferEngine {
  public:
    // models_dir：导出的 .pt 目录；use_cuda：false 时强制 CPU；
    // lazy：true 时每次 run 前加载模型、跑完释放（低显存机器用，显存峰值低）
    InferEngine(const std::string& models_dir, bool use_cuda, bool lazy = false);
    ~InferEngine();

    InferEngine(const InferEngine&) = delete;
    InferEngine& operator=(const InferEngine&) = delete;

    // 执行一次推理，结果写入 out。成功返回 0，失败返回 <0 并填充 err。
    int run(const InferOptions& opt, InferResult& out, std::string& err);

    // CUDA 是否可用（无 NVIDIA GPU 时 false）
    static bool cuda_available();

  private:
    int run_impl(const InferOptions& opt, InferResult& out, std::string& err, int attempt);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// 便捷接口：一次性推理（每次重新加载模型，供 CLI 使用）。
int run_inference(const InferOptions& opt, std::string& err);

}  // namespace ost
