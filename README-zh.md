# OpenSplat3DTrainer

> 单张图片 → 3D 高斯泼溅（3DGS）模型，本地 GPU 推理，无需 Python 运行时。

**English README：[README.md](README.md)**

OpenSplat3DTrainer 是一个本地运行的 3DGS 生成工具：输入一张图片，即可在几十秒内生成一个 3D 高斯泼溅模型，并直接在内置的 3D 视口中查看、导入、导出。

推理内核基于 [TripoSplat](https://github.com/VAST-AI-Research/TripoSplat) 的图像生成方法，用 **LibTorch C++**（TorchScript）重构，无 Python 运行时；前端采用 WebView2 桌面宿主 + 本地 HTTP 服务，3D 视口基于 [Spark](https://github.com/antimatter15/spark) 高斯泼溅渲染器。

---

## 特性

- **单图生成 3DGS**：一张图片生成高质量 3D 高斯模型
- **批量任务队列**：多图批量生成，每个任务可独立设置参数
- **参数自由调节**：采样步数、引导强度、时间偏移、随机种子、高斯数量、输入分辨率、抠图分辨率
- **输入分辨率可调**：512 / 768 / 1024 / 1536，按需平衡速度与细节
- **3D 实时视口**：Spark 渲染，支持三视图 / 网格 / 归位 / 翻转 / 导入 PLY
- **双显存模式**：高显存（模型常驻，性能优先）与低显存（按需加载，省显存）可切换
- **任务进度条**：推理各阶段（抠图 / 特征编码 / 采样步骤 / 高斯解码）实时显示进度
- **日志落盘**：`log.txt` 记录完整运行日志，便于排查问题
- **输出格式**：PLY / SPLAT
- **渲染可调**：内置 3D 视口支持实时调整平滑度与超采样分辨率

---

## 渲染质量对比

内置视口基于 Spark 高斯泼溅渲染，并支持在「渲染」面板中实时调整平滑度与渲染分辨率：

![渲染质量对比](resources/Test.jpg)

---

## 硬件要求

| 项目 | 要求 |
| --- | --- |
| GPU | **NVIDIA 显卡**（CUDA 12+） |
| 显存 | **推荐 12GB 及以上** |
| 最低 | 8GB（低显存模式，按需加载模型） |

- 可用显存 ≥ 10GB：建议使用「高显存模式」，模型常驻显存，速度最快；
- 显存 < 10GB：使用「低显存模式」，按阶段加载/释放模型，峰值显存更低。

> 非 NVIDIA / 无 CUDA 时仍可启动界面，但生成功能不可用。

---

## 下载

- **GitHub Release**：可直接从 [Releases](../../releases) 页面下载最新构建（含 exe、运行库与前端资源）；
- **国内网盘**：Release 说明中同时提供国内网盘分享链接，便于国内用户快速获取；
- 模型权重（`models/`，约 3GB）较大，随 Release 一并提供或在 Release 说明中给出独立下载方式。

---

## 快速开始

1. 将六个模型权重放入 `models/` 目录（来自 TripoSplat 导出）：
   `rmbg.pt`、`dinov3.pt`、`vae_encoder.pt`、`flow_model.pt`、`octree.pt`、`gs_decoder.pt`
2. 将 `output/` 目录中的 `OpenSplat3DTrainer_Launcher.exe` 及相关 DLL 部署到目标机器
3. 双击启动：
   - 首次启动选择显存模式（高 / 低）
   - 「添加图片」或直接把图片拖入窗口
   - 选择任务调整参数（或使用默认参数批量生成）
   - 点击「开始生成」，观察进度条
4. 生成完成后双击任务即可在 3D 视口中查看，或直接导入任意 PLY

---

## 推理管线

```
输入图片
   │  BiRefNet 抠图（分辨率可调）
   ▼
特征编码（DINOv3 + Flux2VAE）
   ▼
Flow 采样（步数 / 引导 / 偏移可调）┄┄ 每步上报进度
   ▼
Octree + GS 解码 → 高斯属性（位置 / 颜色 / 缩放 / 旋转）
   ▼
导出 PLY / SPLAT + 3D 视口预览
```

---

## 构建

- **依赖**：Visual Studio 2022+、CMake 3.20+、LibTorch（CUDA 版，通过 `CMAKE_PREFIX_PATH` 指定）
- **WebView2 SDK**：非 MIT 许可，不入库；`scripts/build.ps1` 会在缺失时自动执行 `scripts/fetch_webview2.ps1` 从 NuGet 拉取（也可手动运行）
- **内核**（`app/`）：推理内核与导出器（LibTorch C++）
- **启动器**（`launcher/`）：WebView2 桌面宿主 + 前端（HTML/CSS/JS + Three.js + Spark）

```powershell
# 示例（以 conda 中的 torch 为前缀）
cmake -S launcher -B launcher/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="<torch路径>"
cmake --build launcher/build --config Release
```

构建产物部署到 `output/` 目录（exe + 运行库 + `web/` 前端 + `models/`）。

## 第三方组件与许可

| 组件 | 许可 | 入库方式 |
| --- | --- | --- |
| [Spark](https://github.com/sparkjsdev/spark) | MIT | 直接入库（`launcher/third_party/Spark`、`launcher/web/vendor/Spark`） |
| [three.js](https://threejs.org/) | MIT | 直接入库（`launcher/web/vendor/three`） |
| [nlohmann/json](https://github.com/nlohmann/json) | MIT | 直接入库（`launcher/third_party/nlohmann`） |
| WebView2 SDK | 微软宽松许可（非 MIT） | 不入库，构建时由脚本从 NuGet 拉取 |
| TripoSplat 模型权重 | 遵循 TripoSplat 上游协议 | 不入库（`models/`），随 Release 提供 |

---

## 未来规划

- **接入图片生成模型**：支持文字 / 参考图生成输入图像，降低对输入图片的依赖；
- **多角度绘图增强**：利用多视角一致性进一步强化生成质量，让 3D 重建更完整、更稳定；
- 更多导出格式与渲染后端。

---

## 致谢

- [Tripo / TripoSplat](https://github.com/VAST-AI-Research/TripoSplat) —— 本项目的图像生成方法与模型权重源自其开源工作；
- [antimatter15 / Spark](https://github.com/antimatter15/spark) —— 3D 高斯泼溅实时渲染器；
- [three.js](https://threejs.org/) —— WebGL 3D 基础库。

---

## 贡献

欢迎一切形式的贡献：Bug 修复、功能建议、文档改进、UI 优化。

- 提交 Issue 描述问题（附上 `output/log.txt` 日志更佳）；
- Fork 并提交 Pull Request；
- 对推理内核（C++ / CUDA）、前端（HTML / JS / 设计）或渲染管线感兴趣，都可以找到合适的切入点。

在贡献之前，请确保你的改动至少满足：**能构建通过**、**不引入新的显存峰值**、**中文界面与日志保持一致**。

---

## License

本项目采用 **MIT License** 开源（见 [LICENSE](LICENSE)）。

其中模型权重与生成方法源自 [TripoSplat](https://github.com/VAST-AI-Research/TripoSplat)，渲染器与前端依赖（Spark、three.js 等）遵循各自上游的开源协议，请在使用时一并遵守。
