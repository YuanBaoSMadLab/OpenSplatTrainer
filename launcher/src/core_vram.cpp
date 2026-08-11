// 显存探测与自适应（不会爆显存）。
#include "core_vram.h"

#include <cstdlib>

#include <torch/script.h>
#include <torch/cuda.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>

namespace ost {

VramInfo vram_probe() {
    VramInfo info;
    if (!torch::cuda::is_available()) {
        return info;
    }
    cudaSetDevice(0);
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        return info;
    }
    info.cuda_available = true;
    info.free_mb = free_b / (1024 * 1024);
    info.total_mb = total_b / (1024 * 1024);
    return info;
}

// 按显卡总显存自动选择高斯数（单位 MiB），保证推理峰值不超预算：
//   8GB 及以下 -> 16384；12GB -> 32768；16GB -> 131072；24GB+ -> 262144
// 注意基于总显存而非当前空闲：低显存模式下模型按需释放会使空闲显存虚高，
// 若按空闲显存计算会自动调高到爆显存的程度。
int auto_num_gaussians(uint64_t total_mb) {
    const uint64_t gb = 1024;
    if (total_mb < 10 * gb) return 16384;
    if (total_mb < 13 * gb) return 32768;
    if (total_mb < 18 * gb) return 131072;
    return 262144;
}

// 按显卡总显存自动选择抠图分辨率：
//   8GB 及以下 -> 512；12GB -> 768；16GB+ -> 1024
int auto_rmbg_res(uint64_t total_mb) {
    const uint64_t gb = 1024;
    if (total_mb < 10 * gb) return 512;
    if (total_mb < 13 * gb) return 768;
    return 1024;
}

bool should_lazy_models(uint64_t free_mb) {
    const uint64_t gb = 1024;
    return free_mb < 10 * gb;
}

void cuda_setup() {
    // CUDA 分配器调优预留（低显存机器可在此扩展配置）
    if (!torch::cuda::is_available()) return;
}

void vram_release_cache() {
    if (!torch::cuda::is_available()) return;
    try {
        at::cuda::CUDACachingAllocator::emptyCache();
    } catch (...) {
    }
}

}  // namespace ost
