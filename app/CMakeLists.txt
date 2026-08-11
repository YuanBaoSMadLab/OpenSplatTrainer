cmake_minimum_required(VERSION 3.20)
project(ostsplat CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# LibTorch 来自 Python 侧 torch 安装（conda），通过 CMAKE_PREFIX_PATH 传入
# 例：cmake -S app -B app/build -DCMAKE_PREFIX_PATH=C:/Users/38820/miniconda3/Lib/site-packages/torch
find_package(Torch REQUIRED)

add_executable(ostsplat
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/infer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ply_writer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/image_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops.cpp
)

target_include_directories(ostsplat PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# TORCH_LIBRARIES 是空格分隔的 plain 列表，单独链接
foreach(lib ${TORCH_LIBRARIES})
    target_link_libraries(ostsplat PRIVATE "${lib}")
endforeach()

# torch 运行依赖 DLL 与 exe 同目录
set_target_properties(ostsplat PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin
)

if(WIN32)
    target_compile_options(ostsplat PRIVATE /utf-8 /W3)
    target_compile_definitions(ostsplat PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
    # 图片解码使用 Windows WIC（无第三方依赖）
    target_link_libraries(ostsplat PRIVATE windowscodecs ole32)
endif()
