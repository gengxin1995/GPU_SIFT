cmake_minimum_required(VERSION 2.8.3)
project(GPU_SIFT)

# OpenCV依赖
find_package( OpenCV REQUIRED )
find_package(CUDA REQUIRED)

# SiftGPU：手动设置其头文件与库文件所在位置
include_directories("/home/xin/Downloads/SiftGPU/src/SiftGPU/" )
set(SIFTGPU_LIBS "/home/xin/Downloads/SiftGPU/bin/libsiftgpu.so")

add_executable( GPU_SIFT main.cpp )

target_link_libraries( GPU_SIFT
    /usr/local/cuda/lib64/libcudadevrt.a 
    -lopencv_core
    -lopencv_highgui
    -lopencv_imgproc
    -lopencv_features2d
    ${SIFTGPU_LIBS}
)
