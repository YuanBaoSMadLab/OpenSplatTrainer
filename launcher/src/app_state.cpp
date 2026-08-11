// 应用共享状态实现。
#include "app_state.h"

#include <algorithm>

namespace ost {

void AppState::add_log(const std::string& msg, bool is_error) {
    std::lock_guard<std::mutex> lk(mtx);
    logs.push_back({msg, is_error});
    if (logs.size() > 5000) {
        logs.erase(logs.begin(), logs.begin() + (int)logs.size() - 5000);
    }
}

std::vector<AppState::LogLine> AppState::recent_logs(size_t n) {
    std::lock_guard<std::mutex> lk(mtx);
    if (logs.size() <= n) return logs;
    return std::vector<LogLine>(logs.end() - (ptrdiff_t)n, logs.end());
}

void AppState::add_task(const TaskParams& p) {
    std::lock_guard<std::mutex> lk(mtx);
    QueueItem it;
    it.params = p;
    queue.push_back(std::move(it));
}

void AppState::clear_tasks() {
    std::lock_guard<std::mutex> lk(mtx);
    queue.clear();
}

}  // namespace ost
