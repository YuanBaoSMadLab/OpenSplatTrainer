cmake_minimum_required(VERSION 3.20)
project(OpenSplat3DTrainer_Launcher CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# LibTorch 来自 Python 侧 torch 安装（conda），通过 CMAKE_PREFIX_PATH 传入
find_package(Torch REQUIRED)

set(APP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../app)
set(TP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
set(WEBVIEW2_DIR ${TP_DIR}/webview2/sdk/build/native)

add_executable(OpenSplat3DTrainer_Launcher WIN32
    src/main.cpp
    src/app_state.cpp
    src/core_task.cpp
    src/core_vram.cpp
    # 复用推理内核（app/）
    ${APP_DIR}/infer.cpp
    ${APP_DIR}/image_utils.cpp
    ${APP_DIR}/ply_writer.cpp
    ${APP_DIR}/ops.cpp
)

target_include_directories(OpenSplat3DTrainer_Launcher PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${TP_DIR}
    ${WEBVIEW2_DIR}/include
    ${APP_DIR}
)

# TORCH_LIBRARIES 是空格分隔的 plain 列表，单独链接
foreach(lib ${TORCH_LIBRARIES})
    target_link_libraries(OpenSplat3DTrainer_Launcher PRIVATE "${lib}")
endforeach()

target_link_libraries(OpenSplat3DTrainer_Launcher PRIVATE
    ${WEBVIEW2_DIR}/x64/WebView2LoaderStatic.lib
    windowscodecs ole32 shell32
)

target_compile_definitions(OpenSplat3DTrainer_Launcher PRIVATE WEBVIEW2_STATIC)

if(WIN32)
    target_compile_options(OpenSplat3DTrainer_Launcher PRIVATE /utf-8 /W3)
    target_compile_definitions(OpenSplat3DTrainer_Launcher PRIVATE
        _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN UNICODE _UNICODE)
endif()
