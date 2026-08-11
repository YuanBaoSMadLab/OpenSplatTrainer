// PLY / SPLAT 写出实现。
#include "ply_writer.h"

#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace ost {

// UTF-8 → UTF-16（输出路径含中文时，必须用宽字符打开文件）
static std::wstring utf8_to_wide(const std::string& s) {
    if (s.empty()) return L"";
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    if (n <= 0) return L"";
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], n);
    w.pop_back();  // 去掉结尾 NUL
    return w;
}

// 前向声明（upright_transform 使用）
static void quat_to_matrix(const float* q, float* m);
static void matrix_to_quat(const float* m, float* q);

// 正立坐标变换：把归一化坐标 [0,1] 转成 y-up 世界坐标（= 官方 viewer 补偿后的方向），
// world = (y-0.5, z-0.5, x-0.5)，对应矩阵行 {0,1,0},{0,0,1},{1,0,0}
static const float kUprightTransform[3][3] = {
    {0, 1, 0},
    {0, 0, 1},
    {1, 0, 0},
};

// 应用正立变换：out_xyz = (y-0.5, z-0.5, x-0.5)，旋转四元数随变换矩阵同步更新
void upright_transform(const float* p, const float* q, float* out_xyz, float* out_rot) {
    out_xyz[0] = p[1] - 0.5f;
    out_xyz[1] = p[2] - 0.5f;
    out_xyz[2] = p[0] - 0.5f;
    float m[9];
    quat_to_matrix(q, m);
    float m2[9];
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            m2[r * 3 + c] = 0;
            for (int k = 0; k < 3; ++k) {
                m2[r * 3 + c] += kUprightTransform[r][k] * m[k * 3 + c];
            }
        }
    }
    matrix_to_quat(m2, out_rot);
}

// 四元数 → 旋转矩阵（对应 _quat_to_matrix）
static void quat_to_matrix(const float* q, float* m /*3x3, row-major*/) {
    float norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (norm < 1e-12f) norm = 1.0f;
    float w = q[0] / norm, x = q[1] / norm, y = q[2] / norm, z = q[3] / norm;
    m[0] = 1 - 2 * (y * y + z * z);
    m[1] = 2 * (x * y - w * z);
    m[2] = 2 * (x * z + w * y);
    m[3] = 2 * (x * y + w * z);
    m[4] = 1 - 2 * (x * x + z * z);
    m[5] = 2 * (y * z - w * x);
    m[6] = 2 * (x * z - w * y);
    m[7] = 2 * (y * z + w * x);
    m[8] = 1 - 2 * (x * x + y * y);
}

// 矩阵 → 四元数（对应 _matrix_to_quat，Shepperd 方法）
static void matrix_to_quat(const float* m /*3x3*/, float* q /*4*/) {
    float trace = m[0] + m[4] + m[8];
    float s = std::sqrt(std::max(trace + 1.0f, 0.0f)) * 2.0f;
    float q0 = 0.25f * s;
    float q1 = (m[7] - m[5]) / (s != 0 ? s : 1.0f);
    float q2 = (m[2] - m[6]) / (s != 0 ? s : 1.0f);
    float q3 = (m[3] - m[1]) / (s != 0 ? s : 1.0f);
    bool m01 = (m[0] >= m[4]) && (m[0] >= m[8]) && (s == 0);
    float s1 = std::sqrt(std::max(1 + m[0] - m[4] - m[8], 0.0f)) * 2.0f;
    if (m01) {
        q0 = (m[7] - m[5]) / (s1 != 0 ? s1 : 1.0f);
        q1 = 0.25f * s1;
        q2 = (m[1] + m[3]) / (s1 != 0 ? s1 : 1.0f);
        q3 = (m[2] + m[6]) / (s1 != 0 ? s1 : 1.0f);
    }
    bool m11 = (m[4] > m[0]) && (m[4] >= m[8]) && (s == 0);
    float s2 = std::sqrt(std::max(1 + m[4] - m[0] - m[8], 0.0f)) * 2.0f;
    if (m11) {
        q0 = (m[2] - m[6]) / (s2 != 0 ? s2 : 1.0f);
        q1 = (m[1] + m[3]) / (s2 != 0 ? s2 : 1.0f);
        q2 = 0.25f * s2;
        q3 = (m[5] + m[7]) / (s2 != 0 ? s2 : 1.0f);
    }
    bool m21 = (m[8] > m[0]) && (m[8] > m[4]) && (s == 0);
    float s3 = std::sqrt(std::max(1 + m[8] - m[0] - m[4], 0.0f)) * 2.0f;
    if (m21) {
        q0 = (m[3] - m[1]) / (s3 != 0 ? s3 : 1.0f);
        q1 = (m[2] + m[6]) / (s3 != 0 ? s3 : 1.0f);
        q2 = (m[5] + m[7]) / (s3 != 0 ? s3 : 1.0f);
        q3 = 0.25f * s3;
    }
    float nq = std::sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
    if (nq < 1e-12f) nq = 1.0f;
    q[0] = q0 / nq; q[1] = q1 / nq; q[2] = q2 / nq; q[3] = q3 / nq;
}

// 将 tensor 拷到 CPU 连续 float 数组
static bool tensor_to_float(const torch::Tensor& t, std::vector<float>& out, int64_t expected, std::string& err) {
    auto c = t.detach().cpu().contiguous();
    if (c.numel() != expected) {
        err = "tensor size mismatch";
        return false;
    }
    if (c.scalar_type() != torch::kFloat32) {
        c = c.to(torch::kFloat32);
    }
    out.assign(c.data_ptr<float>(), c.data_ptr<float>() + c.numel());
    return true;
}

static bool prepare_ply_data(
    const torch::Tensor& xyz_norm,
    const torch::Tensor& features_dc,
    const torch::Tensor& opacity,
    const torch::Tensor& scaling,
    const torch::Tensor& rotation,
    std::vector<float>& xyz,        // [N,3] 世界坐标
    std::vector<float>& f_dc,       // [N,3]
    std::vector<float>& opac,       // [N] 逆激活（logit）
    std::vector<float>& scale,      // [N,3] log
    std::vector<float>& rot,        // [N,4]
    std::string& err) {
    int64_t n = xyz_norm.size(0);
    std::vector<float> xyz_n, fdc_t, op_t, sc_t, rot_t;
    if (!tensor_to_float(xyz_norm, xyz_n, n * 3, err) ||
        !tensor_to_float(features_dc, fdc_t, n * 3, err) ||
        !tensor_to_float(opacity, op_t, n * 1, err) ||
        !tensor_to_float(scaling, sc_t, n * 3, err) ||
        !tensor_to_float(rotation, rot_t, n * 4, err)) {
        return false;
    }

    xyz.resize(n * 3);
    f_dc.resize(n * 3);
    opac.resize(n);
    scale.resize(n * 3);
    rot.resize(n * 4);

    const float aabb_min[3] = {-0.5f, -0.5f, -0.5f};
    const float aabb_size[3] = {1.0f, 1.0f, 1.0f};

    for (int64_t i = 0; i < n; ++i) {
        // 归一化坐标 → aabb 世界坐标
        float p[3] = {
            xyz_n[i * 3 + 0] * aabb_size[0] + aabb_min[0],
            xyz_n[i * 3 + 1] * aabb_size[1] + aabb_min[1],
            xyz_n[i * 3 + 2] * aabb_size[2] + aabb_min[2],
        };

        // 旋转应用默认坐标系变换
        float q[4] = {rot_t[i * 4 + 0], rot_t[i * 4 + 1], rot_t[i * 4 + 2], rot_t[i * 4 + 3]};
        float m[9];
        quat_to_matrix(q, m);
        float m2[9];
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                m2[r * 3 + c] = 0;
                for (int k = 0; k < 3; ++k) {
                    m2[r * 3 + c] += kUprightTransform[r][k] * m[k * 3 + c];
                }
            }
        }
        // xyz 变换：xyz * transform.T
        float px = p[0] * kUprightTransform[0][0] + p[1] * kUprightTransform[0][1] + p[2] * kUprightTransform[0][2];
        float py = p[0] * kUprightTransform[1][0] + p[1] * kUprightTransform[1][1] + p[2] * kUprightTransform[1][2];
        float pz = p[0] * kUprightTransform[2][0] + p[1] * kUprightTransform[2][1] + p[2] * kUprightTransform[2][2];

        xyz[i * 3 + 0] = px;
        xyz[i * 3 + 1] = py;
        xyz[i * 3 + 2] = pz;

        // f_dc
        f_dc[i * 3 + 0] = fdc_t[i * 3 + 0];
        f_dc[i * 3 + 1] = fdc_t[i * 3 + 1];
        f_dc[i * 3 + 2] = fdc_t[i * 3 + 2];

        // opacity 逆激活（logit）＝ log(o/(1-o))
        float o = std::clamp(op_t[i], 1e-6f, 1.0f - 1e-6f);
        opac[i] = std::log(o / (1.0f - o));

        // scaling → log
        scale[i * 3 + 0] = std::log(std::max(sc_t[i * 3 + 0], 1e-8f));
        scale[i * 3 + 1] = std::log(std::max(sc_t[i * 3 + 1], 1e-8f));
        scale[i * 3 + 2] = std::log(std::max(sc_t[i * 3 + 2], 1e-8f));

        // rotation（变换后）
        float q2[4];
        matrix_to_quat(m2, q2);
        rot[i * 4 + 0] = q2[0];
        rot[i * 4 + 1] = q2[1];
        rot[i * 4 + 2] = q2[2];
        rot[i * 4 + 3] = q2[3];
    }
    return true;
}

static bool write_binary_ply(const std::string& path,
                             const std::vector<float>& xyz,
                             const std::vector<float>& f_dc,
                             const std::vector<float>& opac,
                             const std::vector<float>& scale,
                             const std::vector<float>& rot,
                             std::string& err) {
    int64_t n = (int64_t)xyz.size() / 3;

    FILE* f = _wfopen(utf8_to_wide(path).c_str(), L"wb");
    if (!f) {
        err = "cannot open " + path;
        return false;
    }

    // 属性顺序：x,y,z,nx,ny,nz, f_dc_0..2, opacity, scale_0..2, rot_0..3
    std::fprintf(f, "ply\n");
    std::fprintf(f, "format binary_little_endian 1.0\n");
    std::fprintf(f, "element vertex %lld\n", (long long)n);
    std::fprintf(f, "property float x\nproperty float y\nproperty float z\n");
    std::fprintf(f, "property float nx\nproperty float ny\nproperty float nz\n");
    std::fprintf(f, "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n");
    std::fprintf(f, "property float opacity\n");
    std::fprintf(f, "property float scale_0\nproperty float scale_1\nproperty float scale_2\n");
    std::fprintf(f, "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n");
    std::fprintf(f, "end_header\n");

    std::vector<float> row(17);
    for (int64_t i = 0; i < n; ++i) {
        row[0] = xyz[i * 3 + 0];
        row[1] = xyz[i * 3 + 1];
        row[2] = xyz[i * 3 + 2];
        row[3] = 0; row[4] = 0; row[5] = 0;
        row[6] = f_dc[i * 3 + 0];
        row[7] = f_dc[i * 3 + 1];
        row[8] = f_dc[i * 3 + 2];
        row[9] = opac[i];
        row[10] = scale[i * 3 + 0];
        row[11] = scale[i * 3 + 1];
        row[12] = scale[i * 3 + 2];
        row[13] = rot[i * 4 + 0];
        row[14] = rot[i * 4 + 1];
        row[15] = rot[i * 4 + 2];
        row[16] = rot[i * 4 + 3];
        if (std::fwrite(row.data(), sizeof(float), 17, f) != 17) {
            std::fclose(f);
            err = "write failed: " + path;
            return false;
        }
    }
    std::fclose(f);
    return true;
}

bool write_ply(const std::string& path,
               const torch::Tensor& xyz_norm,
               const torch::Tensor& features_dc,
               const torch::Tensor& opacity,
               const torch::Tensor& scaling,
               const torch::Tensor& rotation,
               std::string& err) {
    std::vector<float> xyz, f_dc, opac, scale, rot;
    if (!prepare_ply_data(xyz_norm, features_dc, opacity, scaling, rotation,
                          xyz, f_dc, opac, scale, rot, err)) {
        return false;
    }
    return write_binary_ply(path, xyz, f_dc, opac, scale, rot, err);
}

bool write_splat(const std::string& path,
                 const torch::Tensor& xyz_norm,
                 const torch::Tensor& features_dc,
                 const torch::Tensor& opacity,
                 const torch::Tensor& scaling,
                 const torch::Tensor& rotation,
                 std::string& err) {
    int64_t n = xyz_norm.size(0);
    std::vector<float> xyz_n, fdc_t, op_t, sc_t, rot_t;
    if (!tensor_to_float(xyz_norm, xyz_n, n * 3, err) ||
        !tensor_to_float(features_dc, fdc_t, n * 3, err) ||
        !tensor_to_float(opacity, op_t, n * 1, err) ||
        !tensor_to_float(scaling, sc_t, n * 3, err) ||
        !tensor_to_float(rotation, rot_t, n * 4, err)) {
        return false;
    }

    const float C0 = 0.28209479177387814f;

    struct SplatRec {
        float xyz[3];
        float scale[3];
        uint8_t rgba[4];
        uint8_t rot[4];
    };
    std::vector<SplatRec> recs(n);
    std::vector<float> key(n);

    for (int64_t i = 0; i < n; ++i) {
        // xyz：归一化 → 正立世界坐标（y-up），旋转四元数随变换同步更新
        float p[3] = {xyz_n[i * 3 + 0], xyz_n[i * 3 + 1], xyz_n[i * 3 + 2]};
        float q[4] = {rot_t[i * 4 + 0], rot_t[i * 4 + 1], rot_t[i * 4 + 2], rot_t[i * 4 + 3]};
        float oq[4];
        upright_transform(p, q, recs[i].xyz, oq);

        // scale：.splat 标准格式为 log 空间（Spark 渲染时 exp 恢复）
        recs[i].scale[0] = std::log(std::max(sc_t[i * 3 + 0], 1e-8f));
        recs[i].scale[1] = std::log(std::max(sc_t[i * 3 + 1], 1e-8f));
        recs[i].scale[2] = std::log(std::max(sc_t[i * 3 + 2], 1e-8f));

        float r = std::clamp((fdc_t[i * 3 + 0] * C0 + 0.5f) * 255.0f, 0.0f, 255.0f);
        float g = std::clamp((fdc_t[i * 3 + 1] * C0 + 0.5f) * 255.0f, 0.0f, 255.0f);
        float b = std::clamp((fdc_t[i * 3 + 2] * C0 + 0.5f) * 255.0f, 0.0f, 255.0f);
        float a = std::clamp(op_t[i] * 255.0f, 0.0f, 255.0f);
        recs[i].rgba[0] = (uint8_t)r;
        recs[i].rgba[1] = (uint8_t)g;
        recs[i].rgba[2] = (uint8_t)b;
        recs[i].rgba[3] = (uint8_t)a;

        float qn = std::sqrt(oq[0] * oq[0] + oq[1] * oq[1] + oq[2] * oq[2] + oq[3] * oq[3]);
        if (qn < 1e-12f) qn = 1.0f;
        for (int k = 0; k < 4; ++k) {
            recs[i].rot[k] = (uint8_t)std::clamp(oq[k] / qn * 128.0f + 128.0f, 0.0f, 255.0f);
        }

        key[i] = op_t[i] * (sc_t[i * 3 + 0] * sc_t[i * 3 + 1] * sc_t[i * 3 + 2]);
    }

    // 按 -key 排序（降序）
    std::vector<int64_t> idx(n);
    for (int64_t i = 0; i < n; ++i) idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(), [&](int64_t a, int64_t b) { return key[a] > key[b]; });

    FILE* f = _wfopen(utf8_to_wide(path).c_str(), L"wb");
    if (!f) {
        err = "cannot open " + path;
        return false;
    }
    for (int64_t i = 0; i < n; ++i) {
        if (std::fwrite(&recs[idx[i]], sizeof(SplatRec), 1, f) != 1) {
            std::fclose(f);
            err = "write failed: " + path;
            return false;
        }
    }
    std::fclose(f);
    return true;
}

}  // namespace ost
