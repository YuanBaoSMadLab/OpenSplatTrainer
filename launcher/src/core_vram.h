// 显存探测与自适应（不会爆显存）。
#pragma once

#include <cstdint>

namespace ost {

struct VramInfo {
    bool cuda_available = false;
    uint64_t total_mb = 0;  // 总显存（MiB）
    uint64_t free_mb = 0;   // 当前可用显存（MiB）
};

// 探测当前显存；无 CUDA 时 cuda_available = false。
VramInfo vram_probe();

// 按显卡总显存自动选择高斯数（单位 MiB）：
//   8GB 及以下 -> 16384；12GB -> 32768；16GB -> 131072；24GB+ -> 262144
int auto_num_gaussians(uint64_t total_mb);

// 按显卡总显存自动选择抠图分辨率：
//   8GB 及以下 -> 512；12GB -> 768；16GB+ -> 1024
int auto_rmbg_res(uint64_t total_mb);

// 建议懒加载模型（每次推理前加载、跑完释放，空闲显存回落）。
// 可用显存 >= 10GB 视为高显存：不做优化、模型直接常驻；< 10GB 为低显存：按需加载。
bool should_lazy_models(uint64_t free_mb);

// CUDA 分配器调优预留（低显存机器可在此扩展配置），启动时调用一次
void cuda_setup();

// 回收 CUDA 缓存（每张完成后调用）
void vram_release_cache();

}  // namespace ost
