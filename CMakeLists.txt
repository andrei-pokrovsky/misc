# CMakeLists.txt to build hellocuda.cu
cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wno-unknown-pragmas -Wno-strict-aliasing")
find_package(CUDA QUIET REQUIRED)

add_subdirectory(toolbox)
 
include(${CMAKE_CURRENT_SOURCE_DIR}/toolbox/cmake/Common.cmake)

# Specify binary name and source file to build it from
include_directories(/usr/local/cuda/samples/common/inc/
    /opt/local/include)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib)
cuda_add_executable(test_cublas test_cublas.cc)
target_link_libraries(test_cublas ${CUDA_CUBLAS_LIBRARIES})

cuda_add_executable(cudnn_eval cudnn_eval.cc)
target_link_libraries(cudnn_eval toolbox ${CUDA_CUBLAS_LIBRARIES} cudnn)
