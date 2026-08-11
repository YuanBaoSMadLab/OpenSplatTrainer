// PLY / SPLAT 写出，对应 triposplat.py 的 Gaussian.to_ply_bytes / to_splat_bytes。
#pragma once

#include <string>

#include <torch/script.h>

namespace ost {

// 将 GS 属性写为 PLY（binary_little_endian）。
// xyz_norm:   [N,3] 归一化坐标（0~1）
// features_dc:[N,1,3] SH DC 系数
// opacity:    [N,1] sigmoid 后
// scaling:    [N,3] 激活后
// rotation:   [N,4] 旋转四元数（已加 rots_bias）
bool write_ply(const std::string& path,
               const torch::Tensor& xyz_norm,
               const torch::Tensor& features_dc,
               const torch::Tensor& opacity,
               const torch::Tensor& scaling,
               const torch::Tensor& rotation,
               std::string& err);

// 写为 SPLAT（32 字节/点，按 alpha*scale 体积降序）。
bool write_splat(const std::string& path,
                 const torch::Tensor& xyz_norm,
                 const torch::Tensor& features_dc,
                 const torch::Tensor& opacity,
                 const torch::Tensor& scaling,
                 const torch::Tensor& rotation,
                 std::string& err);

// 正立坐标变换（y-up）：out_xyz = (y-0.5, z-0.5, x-0.5)，
// 旋转四元数随变换矩阵同步更新。供查看器/SPLAT 编码复用，保证导出与渲染方向一致。
void upright_transform(const float* xyz, const float* rot, float* out_xyz, float* out_rot);

}  // namespace ost
