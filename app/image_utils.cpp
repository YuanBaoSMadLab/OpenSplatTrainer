// 图片解码与预处理实现（Windows WIC + 纯 CPU 像素操作）。
#include "image_utils.h"

#include <windows.h>
#include <wincodec.h>
#include <wrl/client.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

using namespace Microsoft::WRL;

namespace ost {

// UTF-8 → UTF-16（Windows 路径必须用宽字符，支持中文等非 ASCII 路径）
static std::wstring utf8_to_wide(const std::string& s) {
    if (s.empty()) return L"";
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    if (n <= 0) return L"";
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], n);
    w.pop_back();  // 去掉结尾 NUL
    return w;
}

// ---------------------------------------------------------------------------
// WIC 解码
// ---------------------------------------------------------------------------
static bool wic_decode(const std::string& path, DecodedImage& out) {
    std::wstring wpath = utf8_to_wide(path);

    ComPtr<IWICImagingFactory> factory;
    if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                                IID_PPV_ARGS(&factory)))) {
        return false;
    }

    ComPtr<IWICBitmapDecoder> decoder;
    if (FAILED(factory->CreateDecoderFromFilename(
            wpath.c_str(), nullptr, GENERIC_READ,
            WICDecodeMetadataCacheOnDemand, &decoder))) {
        return false;
    }

    ComPtr<IWICBitmapFrameDecode> frame;
    if (FAILED(decoder->GetFrame(0, &frame))) {
        return false;
    }

    UINT w = 0, h = 0;
    frame->GetSize(&w, &h);
    if (w == 0 || h == 0) {
        return false;
    }

    WICPixelFormatGUID fmt;
    frame->GetPixelFormat(&fmt);

    // 统一转换到 32bppBGRA
    ComPtr<IWICFormatConverter> converter;
    if (FAILED(factory->CreateFormatConverter(&converter))) {
        return false;
    }
    if (FAILED(converter->Initialize(frame.Get(), GUID_WICPixelFormat32bppBGRA,
                                     WICBitmapDitherTypeNone, nullptr, 0.0,
                                     WICBitmapPaletteTypeCustom))) {
        return false;
    }

    out.width = static_cast<int>(w);
    out.height = static_cast<int>(h);
    out.rgba.resize(static_cast<size_t>(w) * h * 4);
    if (FAILED(converter->CopyPixels(nullptr, w * 4,
                                     static_cast<UINT>(out.rgba.size()),
                                     out.rgba.data()))) {
        out.rgba.clear();
        return false;
    }

    // BGRA -> RGBA
    for (size_t i = 0; i < out.rgba.size(); i += 4) {
        std::swap(out.rgba[i], out.rgba[i + 2]);
    }
    return true;
}

bool decode_image_wic(const std::string& path, DecodedImage& out, std::string& err) {
    HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    bool need_uninit = (hr == S_OK || hr == S_FALSE);
    bool ok = wic_decode(path, out);
    if (need_uninit) {
        CoUninitialize();
    }
    if (!ok) {
        err = "failed to decode image: " + path;
    }
    return ok;
}

// ---------------------------------------------------------------------------
// 缩放
// ---------------------------------------------------------------------------
void resize_nearest(const uint8_t* src, int sw, int sh,
                    uint8_t* dst, int dw, int dh) {
    for (int y = 0; y < dh; ++y) {
        int sy = std::min(sh - 1, y * sh / dh);
        for (int x = 0; x < dw; ++x) {
            int sx = std::min(sw - 1, x * sw / dw);
            const uint8_t* s = src + ((size_t)sy * sw + sx) * 4;
            uint8_t* d = dst + ((size_t)y * dw + x) * 4;
            d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3];
        }
    }
}

void resize_bilinear(const uint8_t* src, int sw, int sh,
                     uint8_t* dst, int dw, int dh) {
    for (int y = 0; y < dh; ++y) {
        float fy = (y + 0.5f) * sh / dh - 0.5f;
        int y0 = std::max(0, (int)std::floor(fy));
        int y1 = std::min(sh - 1, y0 + 1);
        float wy = fy - y0;
        for (int x = 0; x < dw; ++x) {
            float fx = (x + 0.5f) * sw / dw - 0.5f;
            int x0 = std::max(0, (int)std::floor(fx));
            int x1 = std::min(sw - 1, x0 + 1);
            float wx = fx - x0;
            for (int c = 0; c < 4; ++c) {
                float v = src[((size_t)y0 * sw + x0) * 4 + c] * (1 - wx) * (1 - wy)
                        + src[((size_t)y0 * sw + x1) * 4 + c] * wx * (1 - wy)
                        + src[((size_t)y1 * sw + x0) * 4 + c] * (1 - wx) * wy
                        + src[((size_t)y1 * sw + x1) * 4 + c] * wx * wy;
                dst[((size_t)y * dw + x) * 4 + c] = (uint8_t)std::clamp(v, 0.0f, 255.0f);
            }
        }
    }
}

// Lanczos-3 核
static float lanczos3(float x) {
    if (x < 0) x = -x;
    if (x < 1e-5f) return 1.0f;
    if (x >= 3.0f) return 0.0f;
    float px = 3.14159265358979f * x;
    return 3.0f * std::sin(px) * std::sin(px / 3.0f) / (px * px);
}

void resize_lanczos(const uint8_t* src, int sw, int sh,
                    uint8_t* dst, int dw, int dh) {
    const int radius = 3;
    // 预计算每行的源坐标映射
    std::vector<int> src_y0(dh), src_y1(dh);
    std::vector<std::vector<float>> wyv(dh);
    std::vector<int> src_x0(dw), src_x1(dw);
    std::vector<std::vector<float>> wxv(dw);

    auto build_map = [&](int dst_len, int src_len, std::vector<int>& lo,
                         std::vector<int>& hi, std::vector<std::vector<float>>& wv) {
        for (int d = 0; d < dst_len; ++d) {
            float center = (d + 0.5f) * src_len / dst_len - 0.5f;
            int start = (int)std::ceil(center - radius);
            int end = (int)std::floor(center + radius);
            lo[d] = std::max(0, start);
            hi[d] = std::min(src_len - 1, end);
            wv[d].assign(hi[d] - lo[d] + 1, 0.0f);
            float sum = 0;
            for (int s = lo[d]; s <= hi[d]; ++s) {
                float w = lanczos3(center - s);
                wv[d][s - lo[d]] = w;
                sum += w;
            }
            if (sum > 0) {
                for (auto& w : wv[d]) w /= sum;
            }
        }
    };

    build_map(dh, sh, src_y0, src_y1, wyv);
    build_map(dw, sw, src_x0, src_x1, wxv);

    for (int y = 0; y < dh; ++y) {
        for (int x = 0; x < dw; ++x) {
            float acc[4] = {0, 0, 0, 0};
            for (int sy = src_y0[y]; sy <= src_y1[y]; ++sy) {
                float wy = wyv[y][sy - src_y0[y]];
                for (int sx = src_x0[x]; sx <= src_x1[x]; ++sx) {
                    float wx = wxv[x][sx - src_x0[x]];
                    float w = wy * wx;
                    const uint8_t* s = src + ((size_t)sy * sw + sx) * 4;
                    acc[0] += s[0] * w;
                    acc[1] += s[1] * w;
                    acc[2] += s[2] * w;
                    acc[3] += s[3] * w;
                }
            }
            uint8_t* d = dst + ((size_t)y * dw + x) * 4;
            d[0] = (uint8_t)std::clamp(acc[0], 0.0f, 255.0f);
            d[1] = (uint8_t)std::clamp(acc[1], 0.0f, 255.0f);
            d[2] = (uint8_t)std::clamp(acc[2], 0.0f, 255.0f);
            d[3] = (uint8_t)std::clamp(acc[3], 0.0f, 255.0f);
        }
    }
}

// ---------------------------------------------------------------------------
// MinFilter / bbox / 合成
// ---------------------------------------------------------------------------
void min_filter(uint8_t* alpha, int w, int h, int radius) {
    std::vector<uint8_t> tmp((size_t)w * h);
    std::memcpy(tmp.data(), alpha, (size_t)w * h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            uint8_t mn = 255;
            for (int dy = -radius; dy <= radius; ++dy) {
                int yy = y + dy;
                if (yy < 0 || yy >= h) continue;
                for (int dx = -radius; dx <= radius; ++dx) {
                    int xx = x + dx;
                    if (xx < 0 || xx >= w) continue;
                    mn = std::min(mn, tmp[(size_t)yy * w + xx]);
                }
            }
            alpha[(size_t)y * w + x] = mn;
        }
    }
}

void alpha_bbox(const uint8_t* alpha, int w, int h,
                int& x0, int& y0, int& x1, int& y1) {
    x0 = w; y0 = h; x1 = -1; y1 = -1;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (alpha[(size_t)y * w + x] > 0) {
                x0 = std::min(x0, x);
                y0 = std::min(y0, y);
                x1 = std::max(x1, x);
                y1 = std::max(y1, y);
            }
        }
    }
    if (x1 < x0) { x0 = 0; y0 = 0; x1 = w - 1; y1 = h - 1; }
}

void crop_to_black(const uint8_t* src_rgba, int sw, int sh,
                   int x0, int y0, int x1, int y1,
                   uint8_t* dst_rgba, int dw, int dh) {
    // 先拷贝出裁剪区域，再 Lanczos 缩放到画布尺寸
    std::vector<uint8_t> crop((size_t)(x1 - x0 + 1) * (y1 - y0 + 1) * 4);
    int cw = x1 - x0 + 1;
    int ch = y1 - y0 + 1;
    for (int y = 0; y < ch; ++y) {
        for (int x = 0; x < cw; ++x) {
            const uint8_t* s = src_rgba + ((size_t)(y0 + y) * sw + (x0 + x)) * 4;
            uint8_t* d = crop.data() + ((size_t)y * cw + x) * 4;
            d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3];
        }
    }

    std::vector<uint8_t> scaled((size_t)dw * dh * 4);
    resize_lanczos(crop.data(), cw, ch, scaled.data(), dw, dh);

    // 黑底合成：dst = rgb * alpha + black * (1-alpha)，保留 alpha
    for (int y = 0; y < dh; ++y) {
        for (int x = 0; x < dw; ++x) {
            const uint8_t* s = scaled.data() + ((size_t)y * dw + x) * 4;
            uint8_t* d = dst_rgba + ((size_t)y * dw + x) * 4;
            float a = s[3] / 255.0f;
            d[0] = (uint8_t)(s[0] * a);
            d[1] = (uint8_t)(s[1] * a);
            d[2] = (uint8_t)(s[2] * a);
            d[3] = s[3];
        }
    }
}

// ---------------------------------------------------------------------------
// 缩略图（方形 PNG base64，供前端列表预览）
// ---------------------------------------------------------------------------
static std::string b64_encode(const unsigned char* data, size_t len) {
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

bool make_thumbnail(const std::string& path, int size, std::string& out_png_b64) {
    DecodedImage img;
    std::string err;
    if (!decode_image_wic(path, img, err)) {
        return false;
    }

    // 保持比例缩放，长边 = size
    float s = (float)size / (float)std::max(img.width, img.height);
    int nw = std::max(1, (int)(img.width * s));
    int nh = std::max(1, (int)(img.height * s));
    std::vector<uint8_t> resized((size_t)nw * nh * 4);
    resize_lanczos(img.rgba.data(), img.width, img.height, resized.data(), nw, nh);

    // 方形画布居中（透明背景）
    std::vector<uint8_t> canvas((size_t)size * size * 4, 0);
    int ox = (size - nw) / 2, oy = (size - nh) / 2;
    for (int y = 0; y < nh; y++) {
        for (int x = 0; x < nw; x++) {
            const uint8_t* sp = resized.data() + ((size_t)y * nw + x) * 4;
            uint8_t* dp = canvas.data() + ((size_t)(oy + y) * size + (ox + x)) * 4;
            dp[0] = sp[0]; dp[1] = sp[1]; dp[2] = sp[2]; dp[3] = sp[3];
        }
    }

    // WIC 编码 PNG 到内存流
    ComPtr<IWICImagingFactory> factory;
    if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                                IID_PPV_ARGS(&factory)))) {
        return false;
    }
    ComPtr<IStream> stream;
    if (FAILED(CreateStreamOnHGlobal(nullptr, TRUE, &stream))) {
        return false;
    }
    ComPtr<IWICBitmapEncoder> encoder;
    if (FAILED(factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &encoder))) {
        return false;
    }
    if (FAILED(encoder->Initialize(stream.Get(), WICBitmapEncoderNoCache))) {
        return false;
    }
    ComPtr<IWICBitmapFrameEncode> frame;
    if (FAILED(encoder->CreateNewFrame(&frame, nullptr))) {
        return false;
    }
    if (FAILED(frame->Initialize(nullptr))) {
        return false;
    }
    if (FAILED(frame->SetSize((UINT)size, (UINT)size))) {
        return false;
    }
    WICPixelFormatGUID fmt = GUID_WICPixelFormat32bppBGRA;
    if (FAILED(frame->SetPixelFormat(&fmt))) {
        return false;
    }
    // canvas 为 RGBA，PNG 编码用 32bppBGRA 写入 → 先转 BGRA，否则红蓝互换
    std::vector<uint8_t> bgra = canvas;
    for (size_t i = 0; i < bgra.size(); i += 4) {
        std::swap(bgra[i], bgra[i + 2]);
    }
    if (FAILED(frame->WritePixels((UINT)size, (UINT)(size * 4),
                                  (UINT)((size_t)size * size * 4), (BYTE*)bgra.data()))) {
        return false;
    }
    if (FAILED(frame->Commit()) || FAILED(encoder->Commit())) {
        return false;
    }

    STATSTG st = {};
    if (FAILED(stream->Stat(&st, STATFLAG_NONAME))) {
        return false;
    }
    ULONG len = (ULONG)st.cbSize.QuadPart;
    std::vector<unsigned char> png(len);
    LARGE_INTEGER zero = {};
    stream->Seek(zero, STREAM_SEEK_SET, nullptr);
    ULONG read = 0;
    stream->Read(png.data(), len, &read);
    if (read != len) {
        return false;
    }
    out_png_b64 = b64_encode(png.data(), len);
    return true;
}

}  // namespace ost
