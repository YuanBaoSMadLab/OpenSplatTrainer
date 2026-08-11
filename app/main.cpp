// OpenSplat3DTrainer CLI 入口（C++ + LibTorch，全本地）。
// 用法：
//   ostsplat <input.png> --models-dir models --out out.ply [--splat out.splat]
//          [--seed 42] [--steps 20] [--guidance 3.0] [--shift 3.0]
//          [--num-gaussians 262144] [--no-cuda] [--preview out.png]
#include <cstdio>
#include <cstring>
#include <string>

#include "infer.h"

static void print_usage(const char* prog) {
    printf(
        "OpenSplat3DTrainer - 单图 -> 3D Gaussian Splat（本地 TripoSplat）\n"
        "Usage:\n"
        "  %s <input.png> --models-dir <dir> [options]\n"
        "\n"
        "Options:\n"
        "  --out <path>         输出 PLY 路径 (默认 out.ply)\n"
        "  --splat <path>       输出 SPLAT 路径 (可选)\n"
        "  --preview <path>     输出预处理图路径 (可选，调试用)\n"
        "  --seed <int>         随机种子 (默认 42)\n"
        "  --steps <int>        采样步数 (默认 20)\n"
        "  --guidance <float>   CFG 引导强度 (默认 3.0)\n"
        "  --shift <float>      时间表偏移 (默认 3.0)\n"
        "  --num-gaussians <int> 高斯数量 32768~262144 (默认 262144)\n"
        "  --no-cuda            使用 CPU (慢)\n"
        "  -h, --help           显示帮助\n",
        prog);
}

static std::string get_opt(int argc, char** argv, int& i, const std::string& name) {
    if (i + 1 < argc) {
        i++;
        return argv[i];
    }
    fprintf(stderr, "[ostsplat] ERROR: option %s requires a value\n", name.c_str());
    return "";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string input;
    ost::InferOptions opt;
    opt.models_dir = "models";
    opt.output_ply = "out.ply";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--models-dir") opt.models_dir = get_opt(argc, argv, i, arg);
        else if (arg == "--out") opt.output_ply = get_opt(argc, argv, i, arg);
        else if (arg == "--splat") opt.output_splat = get_opt(argc, argv, i, arg);
        else if (arg == "--preview") opt.output_preview = get_opt(argc, argv, i, arg);
        else if (arg == "--seed") opt.seed = std::stoi(get_opt(argc, argv, i, arg));
        else if (arg == "--steps") opt.steps = std::stoi(get_opt(argc, argv, i, arg));
        else if (arg == "--guidance") opt.guidance_scale = std::stof(get_opt(argc, argv, i, arg));
        else if (arg == "--shift") opt.shift = std::stof(get_opt(argc, argv, i, arg));
        else if (arg == "--num-gaussians") opt.num_gaussians = std::stoi(get_opt(argc, argv, i, arg));
        else if (arg == "--no-cuda") opt.use_cuda = false;
        else if (arg[0] != '-') input = arg;
        else {
            fprintf(stderr, "[ostsplat] ERROR: unknown option %s\n", arg.c_str());
            return 1;
        }
    }

    if (input.empty()) {
        fprintf(stderr, "[ostsplat] ERROR: input image required\n");
        print_usage(argv[0]);
        return 1;
    }
    opt.input_image = input;

    std::string err;
    int count = ost::run_inference(opt, err);
    if (count < 0) {
        fprintf(stderr, "[ostsplat] ERROR: %s\n", err.c_str());
        return 1;
    }

    printf("[ostsplat] OK: %d gaussians\n", count);
    if (!opt.output_ply.empty()) printf("[ostsplat] PLY   -> %s\n", opt.output_ply.c_str());
    if (!opt.output_splat.empty()) printf("[ostsplat] SPLAT -> %s\n", opt.output_splat.c_str());
    return 0;
}
