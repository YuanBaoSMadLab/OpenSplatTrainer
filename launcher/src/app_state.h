// 应用共享状态：任务列表、日志、最新模型结果（线程安全）。
#pragma once

#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <torch/script.h>

#include "core_task.h"

namespace ost {

struct QueueItem {
    TaskParams params;
    bool done = false;
    bool ok = false;
    std::string error;
    int64_t num_gaussians = 0;
    double seconds = 0.0;
    std::string out_ply;
};

// 单个已完成模型的 CPU 张量缓存（供 3D 查看器按需拉取）
struct ModelCacheEntry {
    int64_t count = 0;
    std::string name;
    torch::Tensor xyz;       // [N,3] fp32 CPU
    torch::Tensor color;     // [N,3] RGB 0~1
    torch::Tensor opacity;   // [N,1] 0~1
    torch::Tensor scaling;   // [N,3]
    torch::Tensor rotation;  // [N,4]
};

struct AppState {
    // ---- 任务列表（互斥保护，worker 写 / UI 读）----
    std::mutex mtx;
    std::vector<QueueItem> queue;

    // ---- 已生成模型缓存（path -> 模型，互斥保护）----
    std::map<std::string, ModelCacheEntry> models;
    // ---- 最新成功模型的 CPU 张量（互斥保护，查看器默认显示）----
    bool has_model = false;
    int64_t model_count = 0;
    std::string model_name;  // 输入图片文件名
    int model_version = 0;   // 每次更新模型 +1（UI 据此刷新查看器）
    torch::Tensor xyz_cpu;       // [N,3] fp32
    torch::Tensor color_cpu;     // [N,3] RGB 0~1
    torch::Tensor opacity_cpu;   // [N,1] 0~1
    torch::Tensor scaling_cpu;   // [N,3]
    torch::Tensor rotation_cpu;  // [N,4]

    // ---- 日志（互斥保护）----
    struct LogLine {
        std::string text;
        bool error = false;
    };
    std::vector<LogLine> logs;

    // 追加日志（任意线程）
    void add_log(const std::string& msg, bool is_error = false);
    // 取最新 N 行（UI 线程）
    std::vector<LogLine> recent_logs(size_t n);
    // 追加任务（UI 线程）
    void add_task(const TaskParams& p);
    // 清空未完成任务（UI 线程）
    void clear_tasks();
};

}  // namespace ost
