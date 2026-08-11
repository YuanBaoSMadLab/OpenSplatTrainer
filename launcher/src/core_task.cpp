// 后台批处理任务队列实现。
#include "core_task.h"

#include <chrono>
#include <condition_variable>
#include <deque>
#include <thread>

#include "core_vram.h"

namespace ost {

struct TaskQueue::Impl {
    std::string models_dir;
    bool use_cuda = true;
    bool lazy = false;
    VramMode vram_mode = VramMode::Auto;

    std::deque<TaskParams> pending;
    std::mutex mtx;
    std::condition_variable cv;
    std::thread worker;
    bool stop = false;
    bool running = false;

    LogCb log_cb;
    DoneCb done_cb;
    ProgressCb progress_cb;
};

TaskQueue::TaskQueue(const std::string& models_dir, bool use_cuda, bool lazy)
    : impl_(new Impl) {
    impl_->models_dir = models_dir;
    impl_->use_cuda = use_cuda;
    impl_->lazy = lazy;
    // 初始模式按启动探测固定：lazy=true 为低显存按需加载，否则高显存常驻
    impl_->vram_mode = lazy ? VramMode::Lazy : VramMode::Keep;
    impl_->worker = std::thread([this] { worker_loop(); });
}

TaskQueue::~TaskQueue() {
    shutdown();
}

void TaskQueue::shutdown() {
    {
        std::lock_guard<std::mutex> lk(impl_->mtx);
        impl_->stop = true;
    }
    impl_->cv.notify_all();
    if (impl_->worker.joinable()) {
        impl_->worker.join();
    }
}

bool TaskQueue::enqueue(const TaskParams& p) {
    {
        std::lock_guard<std::mutex> lk(impl_->mtx);
        if (impl_->stop) return false;
        impl_->pending.push_back(p);
    }
    impl_->cv.notify_all();
    return true;
}

void TaskQueue::clear_pending() {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    impl_->pending.clear();
}

int TaskQueue::pending_count() const {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    return (int)impl_->pending.size();
}

bool TaskQueue::busy() const {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    return impl_->running;
}

void TaskQueue::set_log_cb(LogCb cb) {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    impl_->log_cb = std::move(cb);
}

void TaskQueue::set_done_cb(DoneCb cb) {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    impl_->done_cb = std::move(cb);
}

void TaskQueue::set_progress_cb(ProgressCb cb) {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    impl_->progress_cb = std::move(cb);
}

void TaskQueue::set_vram_mode(VramMode m) {
    {
        std::lock_guard<std::mutex> lk(impl_->mtx);
        impl_->vram_mode = m;
    }
    // worker 空闲时也会定时醒来（400ms），模式切换即时生效
    impl_->cv.notify_all();
}

bool TaskQueue::current_lazy() const {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    return impl_->lazy;
}

void TaskQueue::worker_loop() {
    // 引擎按显存模式动态构建：Keep 常驻（模型加载一次复用）、Lazy 按需加载/释放
    std::unique_ptr<InferEngine> engine;
    std::string load_err;
    bool engine_lazy = impl_->lazy;

    auto ensure_engine = [&]() -> bool {
        if (engine) return true;
        try {
            engine = std::make_unique<InferEngine>(impl_->models_dir, impl_->use_cuda,
                                                   engine_lazy);
            load_err.clear();
            return true;
        } catch (const std::exception& e) {
            load_err = e.what();
            if (impl_->log_cb) {
                impl_->log_cb("模型加载失败: " + load_err, true);
            }
            return false;
        }
    };

    // 决策当前是否用懒加载：Keep=false、Lazy=true、Auto 按剩余显存
    auto decide_lazy = [&]() -> bool {
        VramMode m;
        {
            std::lock_guard<std::mutex> lk(impl_->mtx);
            m = impl_->vram_mode;
        }
        if (m == VramMode::Keep) return false;
        if (m == VramMode::Lazy) return true;
        VramInfo v = vram_probe();
        return v.cuda_available && should_lazy_models(v.free_mb);
    };

    for (;;) {
        // 显存模式切换：空闲时也响应（wait_for 400ms 定时醒来检查）
        bool want_lazy = decide_lazy();
        if (want_lazy != engine_lazy) {
            engine_lazy = want_lazy;
            engine.reset();
            vram_release_cache();  // 清空 CUDA 分配器缓存，显存立即回落
            {
                std::lock_guard<std::mutex> lk(impl_->mtx);
                impl_->lazy = engine_lazy;
            }
            if (impl_->log_cb) {
                impl_->log_cb(engine_lazy ? "显存模式：模型按需加载/释放（省显存）"
                                          : "显存模式：模型常驻（性能优先）", false);
            }
        }
        // lazy 引擎构造不加载模型（运行前按需加载）；Keep 引擎构造即加载模型常驻
        ensure_engine();

        TaskParams task;
        {
            std::unique_lock<std::mutex> lk(impl_->mtx);
            impl_->running = false;
            impl_->cv.wait_for(lk, std::chrono::milliseconds(400),
                               [this] { return impl_->stop || !impl_->pending.empty(); });
            if (impl_->stop && impl_->pending.empty()) break;
            if (impl_->pending.empty()) continue;
            task = std::move(impl_->pending.front());
            impl_->pending.pop_front();
            impl_->running = true;
        }

        TaskResult res;
        res.image_path = task.image_path;
        res.tag = task.tag;
        if (impl_->log_cb) {
            impl_->log_cb("[开始] " + task.image_path, false);
        }
        if (!engine) {
            res.error = "模型加载失败: " + load_err;
            if (impl_->done_cb) impl_->done_cb(res, nullptr);
            continue;
        }

        // 显存自适应：任务前检查剩余显存，不足则跳过（不崩溃）
        VramInfo vram = vram_probe();
        if (vram.cuda_available && vram.free_mb < 1024) {
            res.error = "显存不足（剩余 " + std::to_string(vram.free_mb) + " MB），已跳过";
            if (impl_->done_cb) impl_->done_cb(res, nullptr);
            continue;
        }

        InferOptions opt;
        opt.models_dir = impl_->models_dir;
        opt.input_image = task.image_path;
        opt.output_ply = task.out_ply;
        opt.output_splat = task.out_splat;
        opt.seed = task.seed;
        opt.steps = task.steps;
        opt.guidance_scale = task.guidance;
        opt.shift = task.shift;
        opt.num_gaussians = task.num_gaussians;
        opt.rmbg_res = task.rmbg_res;
        opt.canvas_res = task.img_res > 0 ? task.img_res : 1024;
        opt.use_cuda = impl_->use_cuda;
        // 进度转发（带任务 tag，供前端区分多任务）
        opt.progress_cb = [this, tag = task.tag](int pct, const std::string& label) {
            TaskProgress p;
            p.tag = tag;
            p.pct = pct;
            p.label = label;
            std::lock_guard<std::mutex> lk(impl_->mtx);
            if (impl_->progress_cb) impl_->progress_cb(p);
        };

        InferResult model;
        std::string err;
        if (impl_->log_cb) {
            VramInfo vb = vram_probe();
            impl_->log_cb("推理前可用显存: " + std::to_string(vb.free_mb / 1024) + " GB", false);
        }
        int rc = engine->run(opt, model, err);
        if (impl_->log_cb) {
            VramInfo va = vram_probe();
            impl_->log_cb("推理后可用显存: " + std::to_string(va.free_mb / 1024) + " GB", false);
        }

        res.ok = (rc == 0);
        res.error = err;
        res.num_gaussians = model.num_gaussians;
        res.seconds = model.seconds;
        res.out_ply = task.out_ply;
        res.out_splat = task.out_splat;

        // 每张完成后回收 CUDA 缓存，避免批处理显存膨胀
        vram_release_cache();

        if (impl_->done_cb) impl_->done_cb(res, res.ok ? &model : nullptr);
    }
}

}  // namespace ost
