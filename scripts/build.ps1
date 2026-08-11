﻿# OpenSplat3DTrainer 一键构建：导出模型（Python 仅此一次，可跳过）→ 编译 C++ 应用 → 复制产物到 output/
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$appDir = Join-Path $root "app"
$launcherDir = Join-Path $root "launcher"
$outDir = Join-Path $root "output"
$modelsDir = Join-Path $root "models"
$buildDir = Join-Path $appDir "build"
$launcherBuild = Join-Path $launcherDir "build"

# LibTorch 路径（复用 Python 侧 torch 安装）
$torchDir = "C:\Users\38820\miniconda3\Lib\site-packages\torch"
if (-not (Test-Path (Join-Path $torchDir "share\cmake\Torch"))) {
    Write-Host "ERROR: LibTorch not found at $torchDir" -ForegroundColor Red
    exit 1
}

# ---------- [1/4] 导出模型（可选：--skip-export） ----------
if ($args -contains "--skip-export") {
    Write-Host "=== [1/4] Skip model export ==="
} else {
    Write-Host "=== [1/4] Export TripoSplat models (Python, one-time) ==="
    python (Join-Path $PSScriptRoot "export_models.py") --triposplat-dir (Join-Path $root "TripoSplat") --out-dir $modelsDir
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# WebView2 SDK（非 MIT，不入库；缺失时从 NuGet 拉取）
$wv2Header = Join-Path $launcherDir "third_party\webview2\sdk\build\native\include\WebView2.h"
if (-not (Test-Path $wv2Header)) {
    Write-Host "=== 拉取 WebView2 SDK ==="
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "fetch_webview2.ps1")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# ---------- [2/4] CMake 配置 ----------
Write-Host "=== [2/4] Configure C++ app ==="
cmake -S $appDir -B $buildDir `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_PREFIX_PATH="$torchDir"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# ---------- [3/4] 编译 ----------
Write-Host "=== [3/4] Build C++ app ==="
cmake --build $buildDir --config Release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$exe = Join-Path $buildDir "bin\Release\ostsplat.exe"

# ---------- [3.5/4] 编译 Launcher (GUI) ----------
Write-Host "=== [3.5/4] Build Launcher (GUI) ==="
cmake -S $launcherDir -B $launcherBuild `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_PREFIX_PATH="$torchDir"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
cmake --build $launcherBuild --config Release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$launcherExe = Join-Path $launcherBuild "Release\OpenSplat3DTrainer_Launcher.exe"

# ---------- [4/4] 复制到 output/ ----------
Write-Host "=== [4/4] Copy artifacts to output/ ==="
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
Copy-Item -Force $exe $outDir
Copy-Item -Force $launcherExe $outDir

# 复制 WebView2 前端资源（HTML/CSS/JS/Three.js）
$webDir = Join-Path $launcherDir "web"
if (Test-Path $webDir) {
    Copy-Item -Recurse -Force $webDir $outDir
}

$torchLib = Join-Path $torchDir "lib"
Get-ChildItem $torchLib -Filter "*.dll" | ForEach-Object {
    Copy-Item -Force $_.FullName $outDir
}
Copy-Item -Force (Join-Path $torchLib "torch_cpu.dll") $outDir

# 复制 VC 运行库（保证未安装 VS 运行库的机器可运行）
foreach ($dll in @("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "concrt140.dll")) {
    $src = Join-Path $env:WINDIR "System32\$dll"
    if (Test-Path $src) { Copy-Item -Force $src $outDir }
}

# 复制模型（可选：默认不复制，运行时 --models-dir 指向 models/）
if ($args -contains "--with-models") {
    Copy-Item -Recurse -Force $modelsDir $outDir
}

Write-Host ""
Write-Host "Build OK. Output:"
Get-ChildItem $outDir | Select-Object -First 15 Name, Length
Write-Host ""
Write-Host "Usage: .\output\ostsplat.exe <image.png> --models-dir models --out out.ply"
