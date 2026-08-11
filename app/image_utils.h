// 图片解码与预处理工具（Windows WIC 解码 + 纯 CPU 像素操作）。
// 对应 triposplat.py 的 preprocess_image（PIL 逻辑的 C++ 复刻）。
#pragma once

#include <string>
#include <vector>

#include <torch/script.h>

namespace ost {

// 解码图片（PNG/JPEG/BMP/GIF/TIFF/WebP），返回 RGBA8（H×W×4，R,G,B,A 各 1 字节）。
struct DecodedImage {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgba;  // length = width*height*4
};

// 用 Windows WIC 解码任意格式图片。
bool decode_image_wic(const std::string& path, DecodedImage& out, std::string& err);

// 生成方形缩略图（保持比例居中，透明背景），编码为 PNG base64，供前端列表预览。
bool make_thumbnail(const std::string& path, int size, std::string& out_png_b64);

// 最近邻缩放 RGBA（用于 1024 画布缩放与裁剪）。
void resize_nearest(const uint8_t* src, int sw, int sh,
                    uint8_t* dst, int dw, int dh);

// 双线性缩放 RGBA。
void resize_bilinear(const uint8_t* src, int sw, int sh,
                     uint8_t* dst, int dw, int dh);

// 高保真缩放（Lanczos-3 近似，PIL.LANCZOS 的等价物）。
void resize_lanczos(const uint8_t* src, int sw, int sh,
                    uint8_t* dst, int dw, int dh);

// 最小滤波（MinFilter），kernel = 2*radius+1，作用于单通道 alpha。
void min_filter(uint8_t* alpha, int w, int h, int radius);

// 计算 alpha > 0 的包围盒。
void alpha_bbox(const uint8_t* alpha, int w, int h,
                int& x0, int& y0, int& x1, int& y1);

// 裁剪并合成黑底。
void crop_to_black(const uint8_t* src_rgba, int sw, int sh,
                   int x0, int y0, int x1, int y1,
                   uint8_t* dst_rgba, int dw, int dh);

}  // namespace ost
