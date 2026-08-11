# 拉取 Microsoft WebView2 SDK（非 MIT 许可，不入库，构建前按需下载）
# 用法：powershell -ExecutionPolicy Bypass -File scripts\fetch_webview2.ps1
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$sdkRoot = Join-Path $root "launcher\third_party\webview2\sdk"
$version = "1.0.4126-prerelease"
$url = "https://www.nuget.org/api/v2/package/Microsoft.Web.WebView2/$version"

$need = Join-Path $sdkRoot "build\native\include\WebView2.h"
if (Test-Path $need) {
    Write-Host "WebView2 SDK 已存在，跳过：$need"
    exit 0
}

Write-Host "下载 WebView2 SDK $version ..."
$nupkg = Join-Path $env:TEMP "webview2-$version.nupkg"
Invoke-WebRequest -Uri $url -OutFile $nupkg -UseBasicParsing

$tmp = Join-Path $env:TEMP "webview2-sdk-$version"
if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
Expand-Archive -Path $nupkg -DestinationPath $tmp -Force

New-Item -ItemType Directory -Force -Path $sdkRoot | Out-Null
# 构建只需要 build/native（include + x64/x86/arm64 的 Loader），其余按需补充
Copy-Item -Recurse -Force (Join-Path $tmp "build") (Join-Path $sdkRoot "build")
Copy-Item -Force (Join-Path $tmp "LICENSE.txt") $sdkRoot
Copy-Item -Force (Join-Path $tmp "Microsoft.Web.WebView2.nuspec") $sdkRoot

Remove-Item -Recurse -Force $tmp
Remove-Item -Force $nupkg
Write-Host "WebView2 SDK 已就绪：$sdkRoot"
