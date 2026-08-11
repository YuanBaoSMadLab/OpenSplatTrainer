// 后台批处理任务队列：单 worker 线程逐张生成。
#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "infer.h"

namespace ost {

struct TaskParams {
    std::string image_path;
    std::string tag;  // 前端任务唯一标识（同一图片可多任务，回调据此匹配）
    int steps = 20;
    float guidance = 3.0f;
    float shift = 3.0f;
    int num_gaussians = 262144;
    int seed = 42;
    int rmbg_res = 1024;  // 抠图分辨率（512/768/1024），低显存时降级
    int img_res = 1024;   // 输入分辨率（512/768/1024/1536），预处理 min 边缩放目标
    std::string out_ply;    // 可空（不写文件）
    std::string out_splat;  // 可空
};

struct TaskResult {
    std::string image_path;
    std::string tag;
    bool ok = false;
    std::string error;
    int64_t num_gaussians = 0;
    double seconds = 0.0;
    std::string out_ply;
    std::string out_splat;
};

// 任务进度（worker 线程回调，pct 0~100）
struct TaskProgress {
    std::string tag;
    int pct = 0;
    std::string label;
};

// 显存模式：模型常驻 vs 按需加载释放
enum class VramMode {
    Auto = 0,  // 按当前可用显存自动决策
    Keep = 1,  // 模型常驻（性能优先，空闲也占显存）
    Lazy = 2,  // 按需加载/释放（省显存，空闲回落）
};

// 批处理队列（线程安全）。日志/完成回调在 worker 线程调用。
class TaskQueue {
  public:
    using LogCb = std::function<void(const std::string& msg, bool is_error)>;
    // done 回调：result 为任务结果；model 为推理结果张量（仅 ok 时非空，
    // 生命周期仅在回调内有效，如需保留请拷贝）
    using DoneCb =
        std::function<void(const TaskResult& result, const InferResult* model)>;
    // 进度回调：每个推理阶段/采样步结束上报一次
    using ProgressCb = std::function<void(const TaskProgress& progress)>;

    TaskQueue(const std::string& models_dir, bool use_cuda, bool lazy = false);
    ~TaskQueue();

    TaskQueue(const TaskQueue&) = delete;
    TaskQueue& operator=(const TaskQueue&) = delete;

    // 入队；返回 false 表示 worker 未启动/停止
    bool enqueue(const TaskParams& p);
    // 清空尚未开始的任务
    void clear_pending();
    // 等待当前任务完成后停止 worker（析构时自动调用）
    void shutdown();

    // 运行时切换显存模式（空闲时立即生效；任务中于任务间隙生效）
    void set_vram_mode(VramMode m);
    // 当前 worker 实际采用的模式（由 Auto 决策结果回写）
    bool current_lazy() const;

    int pending_count() const;
    bool busy() const;  // worker 正在执行任务

    void set_log_cb(LogCb cb);
    void set_done_cb(DoneCb cb);
    void set_progress_cb(ProgressCb cb);

  private:
    void worker_loop();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ost
