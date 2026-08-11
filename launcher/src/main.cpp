// OpenSplat3DTrainer_Launcher：WebView2 桌面应用宿主。
// 前端为 launcher/web/（HTML/CSS/JS + Three.js），通过 WebMessage 与 C++ 通信。
#define WEBVIEW2_STATIC
// winsock2 必须先于 windows.h（冲突）
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <commdlg.h>
#include <shellapi.h>
#include <shobjidl.h>
#include <wrl/client.h>
#include <wrl/implements.h>
#include <wrl/event.h>
#include <webview2.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "nlohmann/json.hpp"
#include "app_state.h"
#include "core_task.h"
#include "core_vram.h"
#include "image_utils.h"
#include "ply_writer.h"

using Microsoft::WRL::Callback;
using Microsoft::WRL::ComPtr;
using json = nlohmann::json;

#pragma comment(lib, "WebView2LoaderStatic.lib")
#pragma comment(lib, "ws2_32.lib")

static HWND g_hwnd = nullptr;
static ComPtr<ICoreWebView2Controller> g_controller;
static ComPtr<ICoreWebView2> g_webview;

static ost::AppState g_state;
static ost::TaskQueue* g_queue = nullptr;
static bool g_cuda_ok = false;
static std::string g_models_dir;
static std::string g_exe_dir;
static DWORD g_ui_thread = 0;

const UINT WM_APP_POST_JS = WM_APP + 1;

// ---------------------------------------------------------------------------
// 文件日志：所有 JS↔C++ 消息、console 输出、关键事件写入 exe 目录 log.txt
// ---------------------------------------------------------------------------
static std::mutex g_log_mtx;
static void file_log(const std::string& msg) {
    if (g_exe_dir.empty()) return;
    std::lock_guard<std::mutex> lk(g_log_mtx);
    std::ofstream f(g_exe_dir + "\\log.txt", std::ios::app);
    if (!f.is_open()) return;
    SYSTEMTIME st;
    GetLocalTime(&st);
    char buf[64];
    sprintf_s(buf, "[%02d:%02d:%02d.%03d] ", st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);
    f << buf << msg << std::endl;
}

// 注入到每个页面的诊断脚本：把 console / 未捕获异常 / 资源加载失败转发给 C++（写入 log.txt）
static const char* kConsoleHookJs = R"JS((function(){try{
  var w=window.chrome&&window.chrome.webview;if(!w)return;
  function fwd(level){return function(){try{
    var p=Array.prototype.map.call(arguments,function(a){
      if(typeof a==="string")return a;
      if(a&&a.stack)return (a.message||String(a));
      try{return JSON.stringify(a);}catch(e){return String(a);}
    });
    w.postMessage({cmd:"__console",level:level,text:p.join(" ").substring(0,2000)});
  }catch(e){}};}
  console.log=fwd("log");console.warn=fwd("warn");console.error=fwd("error");
  window.addEventListener("error",function(e){
    try{w.postMessage({cmd:"__console",level:"error",text:"uncaught: "+(e&&(e.message||(e.error&&e.error.message)))+" @ "+(e&&e.filename)+":"+(e&&e.lineno)});}catch(x){}
  },false);
  window.addEventListener("unhandledrejection",function(e){
    try{w.postMessage({cmd:"__console",level:"error",text:"rejection: "+(e&&e.reason&&(e.reason.message||e.reason))});}catch(x){}
  },false);
  document.addEventListener("error",function(e){
    var t=e.target;
    if(t&&(t.tagName==="SCRIPT"||t.tagName==="LINK"||t.tagName==="IMG")){
      try{w.postMessage({cmd:"__console",level:"error",text:"resource-fail: "+(t.src||t.href)});}catch(x){}
    }
  },true);
}catch(e){}})();
)JS";

// ---------------------------------------------------------------------------
// 本地 HTTP 服务器：WebView2 虚拟主机映射不支持 ES module 加载，
// 改用 127.0.0.1 动态端口托管 launcher/web/ 下的静态资源。
// ---------------------------------------------------------------------------
static std::wstring utf8_to_wstr(const std::string& s);  // 定义见下方工具函数
static SOCKET g_http_sock = INVALID_SOCKET;
static std::string g_http_port;

static std::string http_mime(const std::string& path) {
    std::string ext;
    size_t d = path.find_last_of('.');
    if (d != std::string::npos) ext = path.substr(d);
    if (ext == ".html") return "text/html; charset=utf-8";
    if (ext == ".css") return "text/css; charset=utf-8";
    if (ext == ".js" || ext == ".mjs") return "text/javascript; charset=utf-8";
    if (ext == ".json" || ext == ".map") return "application/json";
    if (ext == ".png") return "image/png";
    if (ext == ".jpg" || ext == ".jpeg") return "image/jpeg";
    if (ext == ".svg") return "image/svg+xml";
    if (ext == ".webp") return "image/webp";
    if (ext == ".wasm") return "application/wasm";
    if (ext == ".ico") return "image/x-icon";
    if (ext == ".woff2" || ext == ".woff") return "font/woff2";
    return "application/octet-stream";
}

static void http_send_all(SOCKET s, const char* data, int len) {
    int sent = 0;
    while (sent < len) {
        int r = send(s, data + sent, len - sent, 0);
        if (r <= 0) break;
        sent += r;
    }
}

static void http_serve_conn(SOCKET s) {
    char buf[8192];
    int n = recv(s, buf, sizeof(buf) - 1, 0);
    if (n <= 0) { closesocket(s); return; }
    buf[n] = 0;
    std::string req(buf);
    std::string path = "/";
    size_t sp = req.find(' ');
    if (sp != std::string::npos) {
        size_t sp2 = req.find(' ', sp + 1);
        if (sp2 != std::string::npos) path = req.substr(sp + 1, sp2 - sp - 1);
    }
    size_t q = path.find('?');
    if (q != std::string::npos) path = path.substr(0, q);
    if (path.empty() || path == "/") path = "/index.html";

    // 转成 web 根目录下的相对路径（反斜杠分隔）；解析 .. 穿越，root 检查兜底
    std::string rel;
    size_t i = 0;
    while (i <= path.size()) {
        size_t j = path.find('/', i);
        if (j == std::string::npos) j = path.size();
        std::string seg = path.substr(i, j - i);
        if (!seg.empty() && seg != ".") {
            if (seg == "..") {
                size_t p = rel.find_last_of('\\');
                if (p == std::string::npos) rel.clear();
                else rel.resize(p);
            } else {
                if (!rel.empty()) rel += '\\';
                rel += seg;
            }
        }
        i = j + 1;
    }
    std::wstring wfull = utf8_to_wstr(g_exe_dir) + L"\\web\\" + utf8_to_wstr(rel);
    wchar_t fullbuf[MAX_PATH];
    GetFullPathNameW(wfull.c_str(), MAX_PATH, fullbuf, nullptr);
    std::wstring root = utf8_to_wstr(g_exe_dir) + L"\\web";
    std::wstring full(fullbuf);
    if (full.rfind(root, 0) != 0) {
        static const char nf[] = "HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        http_send_all(s, nf, (int)strlen(nf));
        closesocket(s);
        return;
    }
    std::ifstream f(full, std::ios::binary);
    if (!f.is_open()) {
        std::string body = "404 Not Found";
        std::string resp = "HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\n"
                           "Content-Length: " + std::to_string(body.size()) +
                           "\r\nConnection: close\r\n\r\n" + body;
        http_send_all(s, resp.data(), (int)resp.size());
        closesocket(s);
        return;
    }
    std::string data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: " + http_mime(path) +
                       "\r\nContent-Length: " + std::to_string(data.size()) +
                       "\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n";
    http_send_all(s, resp.data(), (int)resp.size());
    http_send_all(s, data.data(), (int)data.size());
    closesocket(s);
}

static void http_server_loop() {
    sockaddr_in got = {};
    int glen = sizeof(got);
    if (getsockname(g_http_sock, (sockaddr*)&got, &glen) != 0) return;
    g_http_port = std::to_string(ntohs(got.sin_port));
    file_log("[HTTP] 本地服务器 http://127.0.0.1:" + g_http_port + "/ 已就绪");
    while (true) {
        SOCKET c = accept(g_http_sock, nullptr, nullptr);
        if (c == INVALID_SOCKET) break;
        std::thread(http_serve_conn, c).detach();
    }
}

static bool http_server_start() {
    WSADATA wd;
    if (WSAStartup(MAKEWORD(2, 2), &wd) != 0) return false;
    g_http_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (g_http_sock == INVALID_SOCKET) return false;
    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;  // 动态端口，避免冲突
    if (bind(g_http_sock, (sockaddr*)&addr, sizeof(addr)) != 0) return false;
    if (listen(g_http_sock, SOMAXCONN) != 0) return false;
    std::thread(http_server_loop).detach();
    // 等待端口确定（loop 线程内 getsockname）
    for (int i = 0; i < 200 && g_http_port.empty(); ++i) Sleep(10);
    return !g_http_port.empty();
}

// ---------------------------------------------------------------------------
// 工具函数
// ---------------------------------------------------------------------------
static std::wstring utf8_to_wstr(const std::string& s) {
    if (s.empty()) return L"";
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
    std::wstring w(n, 0);
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), &w[0], n);
    return w;
}

static std::string wstr_to_utf8(const std::wstring& w) {
    if (w.empty()) return "";
    int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s(n, 0);
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), &s[0], n, nullptr, nullptr);
    return s;
}

static std::string base64_encode(const unsigned char* data, size_t len) {
    static const char* tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((len + 2) / 3 * 4);
    for (size_t i = 0; i < len; i += 3) {
        unsigned v = data[i] << 16;
        if (i + 1 < len) v |= data[i + 1] << 8;
        if (i + 2 < len) v |= data[i + 2];
        out += tbl[(v >> 18) & 63];
        out += tbl[(v >> 12) & 63];
        out += (i + 1 < len) ? tbl[(v >> 6) & 63] : '=';
        out += (i + 2 < len) ? tbl[v & 63] : '=';
    }
    return out;
}

static std::vector<uint8_t> base64_decode(const std::string& s) {
    auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    std::vector<uint8_t> out;
    int buf = 0, bits = 0;
    for (char c : s) {
        if (c == '=') break;
        int v = val(c);
        if (v < 0) continue;
        buf = (buf << 6) | v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back((uint8_t)((buf >> bits) & 0xFF));
        }
    }
    return out;
}

static std::string basename(const std::string& p) {
    size_t s = p.find_last_of("/\\");
    return s == std::string::npos ? p : p.substr(s + 1);
}

// ---------------------------------------------------------------------------
// 配置持久化（config.json：首次启动选择显存模式后保存，后续启动直接使用）
// ---------------------------------------------------------------------------
static std::string config_path() {
    return g_exe_dir + "\\config.json";
}

// 读取已保存的显存模式（"high"/"low"）；无配置返回空串
static std::string read_vram_mode_cfg() {
    std::ifstream f(config_path());
    if (!f.is_open()) return "";
    try {
        json j = json::parse(f);
        std::string m = j.value("vram_mode", "");
        return (m == "high" || m == "low") ? m : "";
    } catch (...) {
        return "";
    }
}

// 保存显存模式（供下次启动直接采用）
static void save_vram_mode_cfg(const std::string& mode) {
    try {
        std::ofstream f(config_path());
        if (!f.is_open()) return;
        json j = {{"vram_mode", mode}};
        f << j.dump(2);
    } catch (...) {
    }
}

// ---------------------------------------------------------------------------
// 输出路径解析：相对路径基于 exe 目录（launcher 同目录），并自动创建目录
// ---------------------------------------------------------------------------
// 逐级创建目录（已存在则忽略错误；宽字符支持中文路径）
static void create_dirs(const std::string& dir) {
    std::wstring wdir = utf8_to_wstr(dir);
    std::wstring cur;
    for (size_t i = 0; i < wdir.size(); ++i) {
        cur += wdir[i];
        if (wdir[i] == L'\\' || wdir[i] == L'/') {
            if (cur.size() > 3) CreateDirectoryW(cur.c_str(), nullptr);
        }
    }
    if (!wdir.empty()) CreateDirectoryW(wdir.c_str(), nullptr);
}

// 将相对输出路径解析为 exe 目录下的绝对路径（绝对路径原样返回）
static std::string resolve_output_path(const std::string& p) {
    if (p.empty()) return p;
    if (p.size() >= 2 && p[1] == ':') return p;              // C:\... 绝对
    if (p[0] == '\\' || p[0] == '/') return p;               // 根路径/UNC
    return g_exe_dir + "\\" + p;                             // 相对 → exe 目录
}

// 确保输出文件所在目录存在
static void ensure_output_dir(const std::string& file_path) {
    if (file_path.empty()) return;
    size_t s = file_path.find_last_of("/\\");
    if (s != std::string::npos) {
        create_dirs(file_path.substr(0, s));
    }
}

// 模型缓存条目 → .splat 二进制（32 字节/点，降采样到 max_points，供 Spark 查看器预览）
// 格式：xyz(float32×3) + scale(线性 float32×3) + rgba(uint8×4) + rot(uint8×4，归一化)
// 注意：Spark 的 decodeAntiSplat 直接把 scale 当线性值使用（不做 exp），所以这里不能写 log 空间。
// 坐标与旋转经 upright_transform 转正立（与导出的 PLY/SPLAT 方向一致）
static std::string encode_entry_splat(const ost::ModelCacheEntry& e, int max_points) {
    int64_t n = e.count;
    if (n <= 0) return "";
    int64_t step = std::max<int64_t>(1, n / std::max(1, max_points));
    auto xyz = e.xyz.contiguous();
    auto rgb = e.color.contiguous();
    auto op = e.opacity.contiguous().reshape({-1});
    auto sc = e.scaling.contiguous();
    auto rt = e.rotation.contiguous();
    const float* px = xyz.data_ptr<float>();
    const float* pr = rgb.data_ptr<float>();
    const float* po = op.data_ptr<float>();
    const float* ps = sc.data_ptr<float>();
    const float* pq = rt.data_ptr<float>();
    int64_t out_n = (n + step - 1) / step;
    std::vector<uint8_t> buf;
    buf.reserve((size_t)out_n * 32);
    for (int64_t i = 0; i < n; i += step) {
        uint8_t d[32];
        float oxyz[3], oq[4];
        ost::upright_transform(px + i * 3, pq + i * 4, oxyz, oq);
        std::memcpy(d + 0, &oxyz[0], 4);
        std::memcpy(d + 4, &oxyz[1], 4);
        std::memcpy(d + 8, &oxyz[2], 4);
        float ls[3] = {
            std::max(ps[i * 3 + 0], 1e-8f),
            std::max(ps[i * 3 + 1], 1e-8f),
            std::max(ps[i * 3 + 2], 1e-8f),
        };
        std::memcpy(d + 12, &ls[0], 4);
        std::memcpy(d + 16, &ls[1], 4);
        std::memcpy(d + 20, &ls[2], 4);
        d[24] = (uint8_t)std::clamp(pr[i * 3 + 0] * 255.0f, 0.0f, 255.0f);
        d[25] = (uint8_t)std::clamp(pr[i * 3 + 1] * 255.0f, 0.0f, 255.0f);
        d[26] = (uint8_t)std::clamp(pr[i * 3 + 2] * 255.0f, 0.0f, 255.0f);
        d[27] = (uint8_t)std::clamp(po[i] * 255.0f, 0.0f, 255.0f);
        float qn = std::sqrt(oq[0] * oq[0] + oq[1] * oq[1] + oq[2] * oq[2] + oq[3] * oq[3]);
        if (qn < 1e-12f) qn = 1.0f;
        d[28] = (uint8_t)std::clamp(oq[0] / qn * 128.0f + 128.0f, 0.0f, 255.0f);
        d[29] = (uint8_t)std::clamp(oq[1] / qn * 128.0f + 128.0f, 0.0f, 255.0f);
        d[30] = (uint8_t)std::clamp(oq[2] / qn * 128.0f + 128.0f, 0.0f, 255.0f);
        d[31] = (uint8_t)std::clamp(oq[3] / qn * 128.0f + 128.0f, 0.0f, 255.0f);
        buf.insert(buf.end(), d, d + 32);
    }
    return base64_encode(buf.data(), buf.size());
}

// 图片路径数组 → 缩略图 base64 数组
static json make_thumbs(const std::vector<std::string>& files) {
    json arr = json::array();
    for (auto& f : files) {
        std::string b;
        if (ost::make_thumbnail(f, 96, b)) {
            arr.push_back(b);
        } else {
            arr.push_back("");
        }
    }
    return arr;
}

// ---------------------------------------------------------------------------
// JS 消息：推送（任意线程）/ 发送（UI 线程）
// ---------------------------------------------------------------------------
static void post_js_ui(const std::string& json_str) {
    if (!g_webview) return;
    // 记录到文件（过滤高频 vram 轮询）
    if (json_str.find("\"event\":\"vram\"") == std::string::npos) {
        file_log("[C++→JS] " + json_str.substr(0, 400));
    }
    std::wstring w = utf8_to_wstr(json_str);
    g_webview->PostWebMessageAsJson(w.c_str());
}

static void post_js(const std::string& json_str) {
    if (GetCurrentThreadId() == g_ui_thread) {
        post_js_ui(json_str);
    } else {
        std::string* s = new std::string(json_str);
        PostMessage(g_hwnd, WM_APP_POST_JS, 0, (LPARAM)s);
    }
}

// ---------------------------------------------------------------------------
// 文件选择（Windows 现代 IFileOpenDialog，资源管理器风格）
// ---------------------------------------------------------------------------
static void pick_images(bool multi, std::vector<std::string>& out) {
    ComPtr<IFileOpenDialog> dlg;
    HRESULT hr = CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER,
                                  IID_PPV_ARGS(&dlg));
    if (FAILED(hr)) return;
    COMDLG_FILTERSPEC filters[] = {
        {L"图片文件 (*.png;*.jpg;*.jpeg;*.bmp;*.webp;*.gif)",
         L"*.png;*.jpg;*.jpeg;*.bmp;*.webp;*.gif"},
        {L"所有文件 (*.*)", L"*.*"},
    };
    dlg->SetFileTypes(2, filters);
    if (multi) {
        DWORD opts = 0;
        if (SUCCEEDED(dlg->GetOptions(&opts))) {
            dlg->SetOptions(opts | FOS_ALLOWMULTISELECT);
        }
    }
    dlg->SetTitle(L"选择图片");
    if (FAILED(dlg->Show(g_hwnd))) return;

    if (multi) {
        ComPtr<IShellItemArray> items;
        if (SUCCEEDED(dlg->GetResults(&items))) {
            DWORD n = 0;
            items->GetCount(&n);
            for (DWORD i = 0; i < n; i++) {
                ComPtr<IShellItem> it;
                if (SUCCEEDED(items->GetItemAt(i, &it))) {
                    LPWSTR p = nullptr;
                    if (SUCCEEDED(it->GetDisplayName(SIGDN_FILESYSPATH, &p))) {
                        out.push_back(wstr_to_utf8(p));
                        CoTaskMemFree(p);
                    }
                }
            }
        }
    } else {
        ComPtr<IShellItem> it;
        if (SUCCEEDED(dlg->GetResult(&it))) {
            LPWSTR p = nullptr;
            if (SUCCEEDED(it->GetDisplayName(SIGDN_FILESYSPATH, &p))) {
                out.push_back(wstr_to_utf8(p));
                CoTaskMemFree(p);
            }
        }
    }
}

// 选择单个 PLY 模型文件（现代对话框）
static bool pick_ply_file(std::string& out) {
    ComPtr<IFileOpenDialog> dlg;
    if (FAILED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER,
                                IID_PPV_ARGS(&dlg))))
        return false;
    COMDLG_FILTERSPEC filters[] = {
        {L"PLY 模型 (*.ply)", L"*.ply"},
        {L"所有文件 (*.*)", L"*.*"},
    };
    dlg->SetFileTypes(2, filters);
    dlg->SetTitle(L"导入 PLY 模型");
    if (FAILED(dlg->Show(g_hwnd))) return false;
    ComPtr<IShellItem> it;
    if (FAILED(dlg->GetResult(&it))) return false;
    LPWSTR p = nullptr;
    if (FAILED(it->GetDisplayName(SIGDN_FILESYSPATH, &p))) return false;
    out = wstr_to_utf8(p);
    CoTaskMemFree(p);
    return true;
}

// ---------------------------------------------------------------------------
// JS 请求处理（WebMessageReceived，UI 线程）
// ---------------------------------------------------------------------------
static void handle_web_message(const std::string& msg) {
    try {
        json j = json::parse(msg);
        // 兼容：WebView2 可能把整条消息包成 JSON 字符串
        if (j.is_string()) {
            j = json::parse(j.get<std::string>());
        }
        std::string cmd = j.value("cmd", "");
        // 注入脚本转发的 console 输出：单独记录，避免污染命令日志
        if (cmd == "__console") {
            file_log(std::string("[JS console][") + j.value("level", "?") + "] " + j.value("text", ""));
            return;
        }
        file_log("[JS→C++] cmd=" + cmd + " " + msg.substr(0, 1200));
        if (cmd == "ready") {
            json r = {
                {"event", "init"},
                {"cuda", g_cuda_ok},
                {"modelsDir", g_models_dir},
                {"version", 2},
                {"hasConfig", !read_vram_mode_cfg().empty()},
                {"vramMode", g_queue && g_queue->current_lazy() ? "low" : "high"},
            };
            post_js(r.dump());
        } else if (cmd == "pickImages") {
            post_js("{\"event\":\"log\",\"text\":\"打开文件选择器…\",\"err\":false}");
            std::vector<std::string> files;
            pick_images(j.value("multi", true), files);
            json r = {{"event", "picked"}, {"paths", files}, {"thumbs", make_thumbs(files)}};
            post_js(r.dump());
        } else if (cmd == "addImages") {
            // 拖放/粘贴路径 → 缩略图（跳过文件对话框）
            std::vector<std::string> files;
            if (j.contains("paths")) {
                for (auto& s : j["paths"]) files.push_back(s.get<std::string>());
            }
            if (files.empty()) {
                post_js("{\"event\":\"log\",\"text\":\"拖放未解析到文件路径\",\"err\":true}");
                return;
            }
            json r = {{"event", "picked"}, {"paths", files}, {"thumbs", make_thumbs(files)}};
            post_js(r.dump());
        } else if (cmd == "addImagesData") {
            // 拖放兜底：WebView2 某些环境 File.path 不可用 → 前端传文件内容(base64)，这里落盘到 imported/
            std::string dir = g_exe_dir + "\\imported";
            create_dirs(dir);
            std::vector<std::string> files;
            if (j.contains("files")) {
                for (auto& f : j["files"]) {
                    std::string name = f.value("name", "");
                    std::string b64 = f.value("data", "");
                    auto bytes = base64_decode(b64);
                    if (bytes.empty() || name.empty()) continue;
                    // 清理文件名（防路径穿越/非法字符）
                    std::string safe = basename(name);
                    for (char& c : safe) {
                        if (c == ':' || c == '*' || c == '?' || c == '"' || c == '<' || c == '>' ||
                            c == '|') c = '_';
                    }
                    size_t dot = safe.find_last_of('.');
                    std::string stem = dot == std::string::npos ? safe : safe.substr(0, dot);
                    std::string ext = dot == std::string::npos ? "" : safe.substr(dot);
                    // 重名自动加序号
                    std::string cand = dir + "\\" + safe;
                    int k = 1;
                    while (GetFileAttributesW(utf8_to_wstr(cand).c_str()) != INVALID_FILE_ATTRIBUTES) {
                        cand = dir + "\\" + stem + " (" + std::to_string(k++) + ")" + ext;
                    }
                    FILE* fout = nullptr;
                    // 必须用宽字符打开：fopen_s 走 ANSI 代码页，中文名会乱码，
                    // 后续 decode_image_wic（UTF-8→wide）将找不到文件
                    _wfopen_s(&fout, utf8_to_wstr(cand).c_str(), L"wb");
                    if (fout) {
                        fwrite(bytes.data(), 1, bytes.size(), fout);
                        fclose(fout);
                        files.push_back(cand);
                        file_log("[拖放] 已写入 " + cand + " (" + std::to_string(bytes.size()) + " B)");
                    }
                }
            }
            if (files.empty()) {
                post_js("{\"event\":\"log\",\"text\":\"拖放内容写入失败\",\"err\":true}");
                return;
            }
            json r2 = {{"event", "picked"}, {"paths", files}, {"thumbs", make_thumbs(files)}};
            post_js(r2.dump());
        } else if (cmd == "start") {
            if (!g_queue) {
                post_js("{\"event\":\"log\",\"text\":\"无 CUDA，无法生成\",\"err\":true}");
                return;
            }
            for (auto& t : j["tasks"]) {
                ost::TaskParams p;
                p.image_path = t.value("path", "");
                p.tag = t.value("tag", "");
                p.steps = t.value("steps", 20);
                p.guidance = t.value("guidance", 3.0f);
                p.shift = t.value("shift", 3.0f);
                p.num_gaussians = t.value("num", 262144);
                p.seed = t.value("seed", 42);
                p.rmbg_res = t.value("rmbg_res", 1024);
                p.img_res = t.value("img_res", 1024);
                // 相对路径基于 exe 目录解析，并确保输出目录存在（避免写文件失败）
                p.out_ply = resolve_output_path(t.value("out_ply", ""));
                p.out_splat = resolve_output_path(t.value("out_splat", ""));
                ensure_output_dir(p.out_ply);
                ensure_output_dir(p.out_splat);
                if (p.image_path.empty()) continue;
                g_state.add_task(p);
                g_queue->enqueue(p);
            }
        } else if (cmd == "clearQueue") {
            if (g_queue) g_queue->clear_pending();
            g_state.clear_tasks();
        } else if (cmd == "vram") {
            ost::VramInfo v = ost::vram_probe();
            bool lazy = g_queue ? g_queue->current_lazy()
                                : (v.cuda_available && ost::should_lazy_models(v.free_mb));
            json r = {
                {"event", "vram"},
                {"cuda", v.cuda_available},
                {"free", v.free_mb},
                {"total", v.total_mb},
                {"auto", ost::auto_num_gaussians(v.total_mb)},
                {"rmbg", ost::auto_rmbg_res(v.total_mb)},
                {"lazy", lazy},
            };
            post_js(r.dump());
        } else if (cmd == "setVramMode") {
            std::string mode = j.value("mode", "high");
            if (mode != "low") mode = "high";
            // 判断是否首次配置（保存前）
            bool was_first = read_vram_mode_cfg().empty();
            save_vram_mode_cfg(mode);
            if (g_queue) {
                // 高显存：模型常驻；低显存：按需加载/释放
                g_queue->set_vram_mode(mode == "low" ? ost::VramMode::Lazy
                                                     : ost::VramMode::Keep);
            }
            post_js((json{{"event", "vramModeSaved"},
                          {"mode", mode},
                          {"restart", was_first}}).dump());
            if (was_first) {
                // 首次选择：自动重启一次，让模式真正生效
                std::wstring exe = utf8_to_wstr(g_exe_dir + "\\OpenSplat3DTrainer_Launcher.exe");
                ShellExecuteW(nullptr, L"open", exe.c_str(), nullptr,
                              utf8_to_wstr(g_exe_dir).c_str(), SW_SHOWNORMAL);
                PostMessage(g_hwnd, WM_CLOSE, 0, 0);
            }
        } else if (cmd == "openPly") {
            std::string path;
            if (!pick_ply_file(path)) {
                // 用户取消，无事发生
            } else {
                std::ifstream f(path, std::ios::binary);
                if (!f.is_open()) {
                    post_js((json{{"event", "log"},
                                  {"text", "无法打开 PLY: " + path},
                                  {"err", true}}).dump());
                } else {
                    std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)),
                                              std::istreambuf_iterator<char>());
                    std::string b64 = base64_encode(data.data(), data.size());
                    post_js((json{{"event", "plyLoaded"},
                                  {"name", basename(path)},
                                  {"data", b64}}).dump());
                }
            }
        } else if (cmd == "getModel") {
            std::string path = j.value("path", "");
            std::string b64, name;
            int64_t count = 0;
            bool ok = false;
            {
                std::lock_guard<std::mutex> lk(g_state.mtx);
                if (!path.empty()) {
                    auto it = g_state.models.find(path);
                    if (it != g_state.models.end()) {
                        b64 = encode_entry_splat(it->second, 65536);
                        name = it->second.name;
                        count = it->second.count;
                        ok = true;
                    }
                } else if (g_state.has_model) {
                    ost::ModelCacheEntry e;
                    e.count = g_state.model_count;
                    e.name = g_state.model_name;
                    e.xyz = g_state.xyz_cpu;
                    e.color = g_state.color_cpu;
                    e.opacity = g_state.opacity_cpu;
                    e.scaling = g_state.scaling_cpu;
                    e.rotation = g_state.rotation_cpu;
                    b64 = encode_entry_splat(e, 65536);
                    name = e.name;
                    count = e.count;
                    ok = true;
                }
            }
            if (!ok) {
                post_js("{\"event\":\"modelData\",\"ok\":false}");
                return;
            }
            json r = {{"event", "modelData"}, {"ok", true}, {"name", name}, {"count", count}, {"data", b64}};
            post_js(r.dump());
        }
    } catch (const std::exception& e) {
        post_js((json{{"event", "log"}, {"text", std::string("命令解析失败: ") + e.what()}, {"err", true}}).dump());
    }
}

// ---------------------------------------------------------------------------
// 队列回调 → JS 事件
// ---------------------------------------------------------------------------
static void bind_queue_callbacks() {
    g_queue->set_log_cb([](const std::string& m, bool e) {
        post_js((json{{"event", "log"}, {"text", m}, {"err", e}}).dump());
    });
    g_queue->set_progress_cb([](const ost::TaskProgress& p) {
        post_js((json{{"event", "progress"}, {"tag", p.tag}, {"pct", p.pct}, {"label", p.label}}).dump());
    });
    g_queue->set_done_cb([](const ost::TaskResult& r, const ost::InferResult* m) {
        {
            std::lock_guard<std::mutex> lk(g_state.mtx);
            for (auto& it : g_state.queue) {
                if (!it.done && it.params.image_path == r.image_path) {
                    it.done = true;
                    it.ok = r.ok;
                    it.error = r.error;
                    it.num_gaussians = r.num_gaussians;
                    it.seconds = r.seconds;
                    it.out_ply = r.out_ply;
                    break;
                }
            }
        }
        json j = {
            {"event", "task"},
            {"path", r.image_path},
            {"tag", r.tag},
            {"ok", r.ok},
            {"error", r.error},
            {"count", r.num_gaussians},
            {"seconds", r.seconds},
            {"out", r.out_ply},
        };
        post_js(j.dump());
        // 生成成功但文件导出失败的警告（如目录无权限），模型仍可进视口
        if (r.ok && !r.error.empty()) {
            post_js((json{{"event", "log"}, {"text", r.error}, {"err", true}}).dump());
        }
        if (r.ok && m) {
            try {
                auto xyz = m->xyz.cpu().contiguous().to(torch::kFloat32);
                auto fd = m->features_dc.cpu().contiguous().to(torch::kFloat32);
                auto op = m->opacity.cpu().contiguous().to(torch::kFloat32);
                auto sc = m->scaling.cpu().contiguous().to(torch::kFloat32);
                auto rt = m->rotation.cpu().contiguous().to(torch::kFloat32);
                std::string name = basename(r.image_path);
                ost::ModelCacheEntry entry;
                entry.count = r.num_gaussians;
                entry.name = name;
                entry.xyz = xyz;
                entry.color = (fd * 0.28209479177387814f + 0.5f).clamp(0, 1);
                entry.opacity = op;
                entry.scaling = sc;
                entry.rotation = rt;
                std::lock_guard<std::mutex> lk(g_state.mtx);
                g_state.models[r.image_path] = entry;  // 缓存（双击缩略图可再次查看）
                g_state.has_model = true;
                g_state.model_count = r.num_gaussians;
                g_state.model_name = name;
                g_state.xyz_cpu = entry.xyz;
                g_state.color_cpu = entry.color;
                g_state.opacity_cpu = entry.opacity;
                g_state.scaling_cpu = entry.scaling;
                g_state.rotation_cpu = entry.rotation;
                g_state.model_version++;
            } catch (const std::exception& e) {
                post_js((json{{"event", "log"}, {"text", std::string("模型回传失败: ") + e.what()}, {"err", true}}).dump());
            }
            json mj = {{"event", "model"},
                       {"name", basename(r.image_path)},
                       {"count", r.num_gaussians},
                       {"path", r.image_path}};
            post_js(mj.dump());
        }
    });
}

// ---------------------------------------------------------------------------
// WebView2 初始化
// ---------------------------------------------------------------------------
static void init_webview(HWND hwnd) {
    std::wstring userdata = utf8_to_wstr(g_exe_dir) + L"\\webview2_data";
    auto cb_env = Callback<ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler>(
        [hwnd](HRESULT r, ICoreWebView2Environment* env) -> HRESULT {
            if (FAILED(r)) return r;
            return env->CreateCoreWebView2Controller(
                hwnd,
                Callback<ICoreWebView2CreateCoreWebView2ControllerCompletedHandler>(
                    [](HRESULT r2, ICoreWebView2Controller* c) -> HRESULT {
                        if (FAILED(r2)) return r2;
                        EventRegistrationToken token_ = {};
                        EventRegistrationToken token_nav = {};
                        g_controller = c;
                        // 必须设置初始尺寸，否则渲染区域为 0（白屏）
                        RECT b;
                        GetClientRect(g_hwnd, &b);
                        c->put_Bounds(b);
                        // 允许 WebView2 接收文件拖放：JS 端用 File.path（WebView2 扩展）取路径，
                        // 比 WM_DROPFILES 更可靠（WM_DROPFILES 保留作兜底）
                        ComPtr<ICoreWebView2Controller4> c4;
                        if (SUCCEEDED(g_controller.As(&c4)) && c4) {
                            c4->put_AllowExternalDrop(TRUE);
                        }
                        c->get_CoreWebView2(&g_webview);
                        // 捕获 JS console / 未捕获异常 / 资源加载失败 → 写入 log.txt。
                        // 本 SDK 版本无 add_ConsoleMessage，改用注入脚本把 console 转发给 C++。
                        g_webview->AddScriptToExecuteOnDocumentCreated(
                            utf8_to_wstr(kConsoleHookJs).c_str(),
                            Callback<ICoreWebView2AddScriptToExecuteOnDocumentCreatedCompletedHandler>(
                                [](HRESULT, LPCWSTR) -> HRESULT { return S_OK; }).Get());
                        // 记录所有非 200 的资源响应（诊断 module / 文件加载失败）
                        ComPtr<ICoreWebView2_2> wv2;
                        if (SUCCEEDED(g_webview.As(&wv2)) && wv2) {
                            EventRegistrationToken token_net = {};
                            wv2->add_WebResourceResponseReceived(
                                Callback<ICoreWebView2WebResourceResponseReceivedEventHandler>(
                                    [](ICoreWebView2*, ICoreWebView2WebResourceResponseReceivedEventArgs* a) -> HRESULT {
                                        ComPtr<ICoreWebView2WebResourceRequest> req;
                                        ComPtr<ICoreWebView2WebResourceResponseView> resp;
                                        if (FAILED(a->get_Request(&req)) || !req) return S_OK;
                                        if (FAILED(a->get_Response(&resp)) || !resp) return S_OK;
                                        LPWSTR u = nullptr;
                                        req->get_Uri(&u);
                                        int status = 0;
                                        resp->get_StatusCode(&status);
                                        if (u) {
                                            if (status != 200) {
                                                file_log("[网络] HTTP " + std::to_string(status) + " " + wstr_to_utf8(u));
                                            }
                                            CoTaskMemFree(u);
                                        }
                                        return S_OK;
                                    }).Get(), &token_net);
                        }
                        // 导航失败诊断（写入 exe 目录 nav_error.txt）
                        g_webview->add_NavigationCompleted(
                            Callback<ICoreWebView2NavigationCompletedEventHandler>(
                                [](ICoreWebView2*, ICoreWebView2NavigationCompletedEventArgs* a) -> HRESULT {
                                    BOOL ok = FALSE;
                                    a->get_IsSuccess(&ok);
                                    if (!ok) {
                                        file_log("[C++→] 页面导航失败");
                                        FILE* f = nullptr;
                                        fopen_s(&f, (g_exe_dir + "\\nav_error.txt").c_str(), "w");
                                        if (f) {
                                            fprintf(f, "WebView2 navigation failed\n");
                                            fclose(f);
                                        }
                                    } else {
                                        // 诊断：动态 import viewer.js（注意 ./ 前缀），捕获具体报错信息
                                        const wchar_t* diag =
                                            L"import('./js/viewer.js').then(function(){if(window.chrome&&window.chrome.webview)"
                                            L"window.chrome.webview.postMessage({event:'log',text:'DYNAMPORT viewer.js OK',err:false});})"
                                            L".catch(function(e){if(window.chrome&&window.chrome.webview)"
                                            L"window.chrome.webview.postMessage({event:'log',text:'DYNAMPORT viewer.js FAIL: '+(e&&e.message),err:true});});";
                                        g_webview->ExecuteScript(diag, nullptr);
                                    }
                                    return S_OK;
                                }).Get(), &token_nav);
                        // 虚拟主机映射（ICoreWebView2_3+）
                        ComPtr<ICoreWebView2_3> wv3;
                        if (SUCCEEDED(g_webview.As(&wv3)) && wv3) {
                            std::wstring root = utf8_to_wstr(g_exe_dir) + L"\\web";
                            wv3->SetVirtualHostNameToFolderMapping(
                                L"app.local", root.c_str(),
                                COREWEBVIEW2_HOST_RESOURCE_ACCESS_KIND_ALLOW);
                        }
                        g_webview->add_WebMessageReceived(
                            Callback<ICoreWebView2WebMessageReceivedEventHandler>(
                                [](ICoreWebView2*, ICoreWebView2WebMessageReceivedEventArgs* a) -> HRESULT {
                                    LPWSTR m = nullptr;
                                    a->get_WebMessageAsJson(&m);
                                    if (m) {
                                        handle_web_message(wstr_to_utf8(m));
                                        CoTaskMemFree(m);
                                    }
                                    return S_OK;
                                }).Get(), &token_);
                        // 本地 HTTP 服务器托管页面（虚拟主机映射不支持 ES module）
                        if (!g_http_port.empty()) {
                            std::wstring url = L"http://127.0.0.1:" + utf8_to_wstr(g_http_port) + L"/index.html";
                            file_log("[C++→] 导航 " + wstr_to_utf8(url));
                            g_webview->Navigate(url.c_str());
                        } else {
                            g_webview->Navigate(L"https://app.local/index.html");
                        }
                        return S_OK;
                    }).Get());
        });
    CreateCoreWebView2EnvironmentWithOptions(nullptr, userdata.c_str(), nullptr,
                                              cb_env.Get());
}

// ---------------------------------------------------------------------------
// 窗口
// ---------------------------------------------------------------------------
static LRESULT WINAPI WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
        case WM_APP_POST_JS: {
            std::string* s = (std::string*)lp;
            post_js_ui(*s);
            delete s;
            return 0;
        }
        case WM_DROPFILES: {
            HDROP drop = (HDROP)wp;
            UINT n = DragQueryFileW(drop, 0xFFFFFFFF, nullptr, 0);
            std::vector<std::string> paths;
            for (UINT i = 0; i < n; i++) {
                wchar_t p[MAX_PATH];
                if (DragQueryFileW(drop, i, p, MAX_PATH)) {
                    paths.push_back(wstr_to_utf8(p));
                }
            }
            DragFinish(drop);
            if (!paths.empty()) {
                post_js((json{{"event", "dropped"}, {"paths", paths},
                              {"thumbs", make_thumbs(paths)}}).dump());
            }
            return 0;
        }
        case WM_SIZE:
            if (g_controller) {
                RECT b;
                GetClientRect(hwnd, &b);
                g_controller->put_Bounds(b);
            }
            return 0;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

static std::string exe_dir() {
    wchar_t buf[MAX_PATH];
    GetModuleFileNameW(nullptr, buf, MAX_PATH);
    std::wstring w(buf);
    size_t p = w.find_last_of(L"\\/");
    return wstr_to_utf8(w.substr(0, p));
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int) {
    g_ui_thread = GetCurrentThreadId();
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    g_cuda_ok = ost::InferEngine::cuda_available();
    ost::cuda_setup();
    g_exe_dir = exe_dir();
    g_models_dir = g_exe_dir + "\\models";
    // 默认输出目录：launcher 同目录 output/（不存在则创建）
    create_dirs(g_exe_dir + "\\output");

    WNDCLASSEXW wc = {};
    wc.cbSize = sizeof(wc);
    wc.style = CS_CLASSDC;
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = L"OpenSplat3DTrainerLauncher";
    wc.hIcon = LoadIcon(nullptr, IDI_APPLICATION);
    RegisterClassExW(&wc);

    g_hwnd = CreateWindowExW(0, wc.lpszClassName, L"OpenSplat3DTrainer_Launcher",
                             WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 1280, 800,
                             nullptr, nullptr, hInstance, nullptr);
    if (!g_hwnd) return 1;
    DragAcceptFiles(g_hwnd, TRUE);
    ShowWindow(g_hwnd, SW_SHOWDEFAULT);
    UpdateWindow(g_hwnd);

    // 先创建任务队列（worker 按显存模式决策模型加载），再初始化 WebView2
    // 虚拟主机映射不支持 ES module，页面由本地 HTTP 服务器托管（http://127.0.0.1）
    http_server_start();

    if (g_cuda_ok) {
        // 优先使用已保存配置；首次启动无配置则按探测：可用显存 >= 10GB 高显存常驻，否则低显存按需
        std::string saved = read_vram_mode_cfg();
        bool lazy;
        if (!saved.empty()) {
            lazy = (saved == "low");
        } else {
            ost::VramInfo v0 = ost::vram_probe();
            lazy = v0.cuda_available && ost::should_lazy_models(v0.free_mb);
        }
        g_queue = new ost::TaskQueue(g_models_dir, true, lazy);
        bind_queue_callbacks();
        g_state.add_log(lazy ? "低显存模式：模型按需加载/释放（空闲显存回落）"
                             : "高显存模式：模型常驻（性能优先）");
    }

    init_webview(g_hwnd);

    MSG msg;
    while (GetMessage(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    if (g_queue) {
        g_queue->shutdown();
        delete g_queue;
        g_queue = nullptr;
    }
    g_webview.Reset();
    g_controller.Reset();
    DestroyWindow(g_hwnd);
    UnregisterClassW(wc.lpszClassName, wc.hInstance);
    return 0;
}
